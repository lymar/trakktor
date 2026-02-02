use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
};

use boa_engine::{
    Context, JsResult, Script, Source,
    context::{ContextBuilder, time::JsInstant},
    job::{
        GenericJob, Job, JobExecutor, NativeAsyncJob, PromiseJob, TimeoutJob,
    },
    property::Attribute,
};
use boa_runtime::{Console, interval};
use futures_concurrency::future::FutureGroup;
use futures_lite::{StreamExt, future};
use tokio::task;

mod config;
pub use config::EngineConfig;

use crate::trk::Trk;
mod api;
mod js_logger;

pub(crate) async fn run(trk_inst: Arc<Trk>) -> anyhow::Result<()> {
    let queue = Rc::new(Queue::new());
    let context = &mut ContextBuilder::new()
        .job_executor(queue.clone())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build context: {e}"))?;
    add_runtime(&trk_inst, context)?;
    self::api::register(context)?;

    let js_code = std::fs::read_to_string(&trk_inst.cfg.js_file)?;

    let local_set = &mut task::LocalSet::default();
    let engine = local_set.run_until(async {
        let script = Script::parse(
            Source::from_bytes(js_code.as_bytes()),
            None,
            context,
        )?;

        script.evaluate_async(context).await?;
        queue.run_jobs_async(&RefCell::new(context)).await
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
    // let console = Console::init(context);
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
