use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
};

use boa_engine::{
    Context, JsArgs, JsData, JsError, JsNativeError, JsResult, JsValue, Module,
    NativeFunction, Source,
    builtins::promise::PromiseState,
    context::{ContextBuilder, time::JsInstant},
    job::{
        GenericJob, Job, JobExecutor, NativeAsyncJob, PromiseJob, TimeoutJob,
    },
    js_string,
    module::SimpleModuleLoader,
    property::Attribute,
};
use boa_gc::{Finalize, Trace};
use boa_runtime::{Console, interval};
use futures_concurrency::future::FutureGroup;
use futures_lite::{StreamExt, future};
use tokio::task;

mod config;
pub use config::EngineConfig;

use crate::trk::Trk;
mod api;
mod js_logger;

#[derive(Trace, Finalize, JsData)]
pub(crate) struct JsTrkHandle {
    #[unsafe_ignore_trace]
    pub trk: Arc<Trk>,
}

pub(crate) async fn run(trk_inst: Arc<Trk>) -> anyhow::Result<()> {
    let js_file = trk_inst.cfg.js_file.canonicalize()?;
    let js_root = js_file.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "JavaScript entry file path has no parent directory: {}",
            js_file.display()
        )
    })?;
    let js_file_name = js_file
        .clone()
        .file_name()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "JavaScript entry file path has no file name: {}",
                js_file.display()
            )
        })?
        .to_owned();

    let loader = Rc::new(SimpleModuleLoader::new(js_root).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create JavaScript module loader (root: {}): {e}",
            js_root.display()
        )
    })?);

    let queue = Rc::new(Queue::new());
    let context = &mut ContextBuilder::new()
        .job_executor(queue.clone())
        .module_loader(loader.clone())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build context: {e}"))?;
    add_runtime(&trk_inst, context)?;
    self::api::register(context)?;

    let realm = context.realm().clone();
    realm.host_defined_mut().insert(JsTrkHandle {
        trk: Arc::clone(&trk_inst),
    });

    let js_code = tokio::fs::read_to_string(&js_file).await?;

    let local_set = &mut task::LocalSet::default();

    let root_module_js_file = format!("./{}", js_file_name.display());

    let engine = local_set.run_until(async {
        let source = Source::from_reader(
            js_code.as_bytes(),
            Some(std::path::Path::new(&root_module_js_file)),
        );
        let module = Module::parse(source, None, context)?;

        loader.insert(js_file, module.clone());

        let module_load_promise = module.load_link_evaluate(context);

        queue.clone().run_jobs_async(&RefCell::new(context)).await?;

        match module_load_promise.state() {
            PromiseState::Pending => {
                return Err(JsNativeError::typ()
                    .with_message(
                        "Root module evaluation did not complete (promise is \
                         still pending)",
                    )
                    .into());
            },
            PromiseState::Fulfilled(_v) => {
                log::debug!("Root module loaded, linked, and evaluated");
            },
            PromiseState::Rejected(err) => {
                return Err(JsError::from_opaque(err)
                    .try_native(context)
                    .map_err(|e| {
                        log::error!(
                            "Failed to convert module rejection into a native \
                             error: {e}"
                        );
                        JsNativeError::typ().with_message(
                            "Module evaluation failed (rejected promise)",
                        )
                    })?
                    .into());
            },
        }

        let namespace = module.namespace(context);
        let run_fn = namespace
            .get(js_string!("run"), context)?
            .as_callable()
            .ok_or_else(|| {
            JsNativeError::typ().with_message(
                "Root module must export a callable `run` function",
            )
        })?;

        let run_fn_promise = run_fn
            .call(&JsValue::undefined(), &[], context)?
            .as_promise()
            .ok_or_else(|| {
                JsNativeError::typ().with_message(
                    "`run()` must return a Promise (did you forget to mark it \
                     async?)",
                )
            })?;

        let run_res_promise = run_fn_promise
            .then(
                Some(
                    NativeFunction::from_fn_ptr(|_, args, _context| {
                        let value = args.get_or_undefined(0);
                        Ok(value.clone())
                    })
                    .to_js_function(context.realm()),
                ),
                Some(
                    NativeFunction::from_fn_ptr(|_, args, _context| {
                        let error = args.get_or_undefined(0);
                        let msg = format!("{}", error.display());
                        log::error!(
                            msg:? = msg;
                            "`run()` rejected",
                        );
                        Err(JsError::from_opaque(error.clone()))
                    })
                    .to_js_function(context.realm()),
                ),
                context,
            )
            .finally(
                NativeFunction::from_fn_ptr(|_, _, _| {
                    log::debug!("`run()` finished (resolved or rejected)");
                    Ok(JsValue::undefined())
                })
                .to_js_function(context.realm()),
                context,
            );

        queue.run_jobs_async(&RefCell::new(context)).await?;

        match run_res_promise.state() {
            PromiseState::Fulfilled(v) => {
                if let Ok(Some(jv)) = v.to_json(context) {
                    if let Ok(j) = serde_json::to_string_pretty(&jv) {
                        println!("{}", j);
                    }
                }
            },
            PromiseState::Rejected(err) => {
                return Err(JsError::from_opaque(err));
            },
            _ => {},
        }

        Result::<(), JsError>::Ok(())
    });

    engine
        .await
        .map_err(|e| anyhow::anyhow!("Engine error: {e}"))?;

    Ok(())
}

fn add_runtime(
    trk_inst: &Arc<Trk>,
    context: &mut Context,
) -> anyhow::Result<()> {
    let console =
        Console::init_with_logger(js_logger::JsLogger::new(trk_inst), context);
    context
        .register_global_property(Console::NAME, console, Attribute::all())
        .map_err(|e| anyhow::anyhow!("Failed to register console: {e}"))?;

    interval::register(context).map_err(|e| {
        anyhow::anyhow!("Failed to register interval functions: {e}")
    })?;

    Ok(())
}

struct Queue {
    async_jobs: RefCell<VecDeque<NativeAsyncJob>>,
    promise_jobs: RefCell<VecDeque<PromiseJob>>,
    timeout_jobs: RefCell<BTreeMap<JsInstant, TimeoutJob>>,
    generic_jobs: RefCell<VecDeque<GenericJob>>,
}

impl Queue {
    fn new() -> Self {
        Self {
            async_jobs: RefCell::default(),
            promise_jobs: RefCell::default(),
            timeout_jobs: RefCell::default(),
            generic_jobs: RefCell::default(),
        }
    }

    fn drain_timeout_jobs(&self, context: &mut Context) {
        let now = context.clock().now();

        let mut timeouts_borrow = self.timeout_jobs.borrow_mut();
        let mut jobs_to_keep = timeouts_borrow.split_off(&now);
        jobs_to_keep.retain(|_, job| !job.is_cancelled());
        let jobs_to_run =
            std::mem::replace(timeouts_borrow.deref_mut(), jobs_to_keep);
        drop(timeouts_borrow);

        for job in jobs_to_run.into_values() {
            if let Err(e) = job.call(context) {
                eprintln!("Uncaught {e}");
            }
        }
    }

    fn drain_jobs(&self, context: &mut Context) {
        self.drain_timeout_jobs(context);

        let job = self.generic_jobs.borrow_mut().pop_front();
        if let Some(generic) = job &&
            let Err(err) = generic.call(context)
        {
            eprintln!("Uncaught {err}");
        }

        let jobs = std::mem::take(&mut *self.promise_jobs.borrow_mut());
        for job in jobs {
            if let Err(e) = job.call(context) {
                eprintln!("Uncaught {e}");
            }
        }
        context.clear_kept_objects();
    }
}

impl JobExecutor for Queue {
    fn enqueue_job(self: Rc<Self>, job: Job, context: &mut Context) {
        match job {
            Job::PromiseJob(job) => {
                self.promise_jobs.borrow_mut().push_back(job)
            },
            Job::AsyncJob(job) => self.async_jobs.borrow_mut().push_back(job),
            Job::TimeoutJob(t) => {
                let now = context.clock().now();
                self.timeout_jobs.borrow_mut().insert(now + t.timeout(), t);
            },
            Job::GenericJob(g) => self.generic_jobs.borrow_mut().push_back(g),
            _ => panic!("unsupported job type"),
        }
    }

    fn run_jobs(self: Rc<Self>, context: &mut Context) -> JsResult<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();

        task::LocalSet::default()
            .block_on(&runtime, self.run_jobs_async(&RefCell::new(context)))
    }

    async fn run_jobs_async(
        self: Rc<Self>,
        context: &RefCell<&mut Context>,
    ) -> JsResult<()> {
        let mut group = FutureGroup::new();
        loop {
            for job in std::mem::take(&mut *self.async_jobs.borrow_mut()) {
                group.insert(job.call(context));
            }

            if group.is_empty() &&
                self.promise_jobs.borrow().is_empty() &&
                self.timeout_jobs.borrow().is_empty() &&
                self.generic_jobs.borrow().is_empty()
            {
                return Ok(());
            }

            if let Some(Err(err)) =
                future::poll_once(group.next()).await.flatten()
            {
                eprintln!("Uncaught {err}");
            };

            self.drain_jobs(&mut context.borrow_mut());
            task::yield_now().await
        }
    }
}
