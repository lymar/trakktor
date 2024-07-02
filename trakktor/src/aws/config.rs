use aws_config::SdkConfig;

pub trait AwsConfigProvider {
    fn get_aws_config(&self) -> &SdkConfig;
}

pub trait CloudFormationStackProvider {
    fn get_stack_prefix(&self) -> &str;

    fn get_base_stack_name(&self) -> Box<str> {
        format!("{}-base", self.get_stack_prefix()).into()
    }

    fn get_gpu_batch_stack_name(&self) -> Box<str> {
        format!("{}-gpu-batch", self.get_stack_prefix()).into()
    }
}

pub trait S3Provider {
    fn get_bucket_name(&self) -> &str;
}
