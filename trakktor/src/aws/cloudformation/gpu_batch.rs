use askama::Template;
use serde::Deserialize;

use super::base::gen_subnet_names;
use crate::whisper;

#[derive(Template)]
#[template(path = "cloudformation/gpu_batch.yaml", escape = "none")]
struct GpuBatchTemplate<'a, T: std::fmt::Display> {
    subnets: &'a [T],
    base_stack_name: &'a str,
    whisper_large_image_name: &'a str,
}

pub fn gen_gpu_batch_template(
    availability_zone_count: usize,
    base_stack_name: &str,
    is_dev: bool,
) -> Box<str> {
    GpuBatchTemplate {
        subnets: &gen_subnet_names(availability_zone_count),
        base_stack_name,
        whisper_large_image_name: &whisper::make_image_name(
            whisper::Model::Large,
            is_dev,
        ),
    }
    .render()
    .expect("Failed to generate template")
    .into()
}

#[test]
fn template_verification_test() {
    let stack = gen_gpu_batch_template(3, "trakktor-net", true);
    println!("{}", stack);

    assert_eq!(
        crate::hasher::get_hash_value(stack.as_bytes()),
        "U8gmQTMfHG_94BaZQAhgqzeukCMdyOzlhDqRP0ahrBk"
    )
}

#[derive(Debug, Deserialize)]
pub struct GpuBatchStackOutputs {
    #[serde(rename = "GpuJobQueue")]
    pub job_queue: String,
    #[serde(rename = "GpuWhisperLargeJob")]
    pub whisper_large_job: String,
}
