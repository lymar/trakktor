use askama::Template;

#[derive(Template)]
#[template(path = "cloudformation/base.yaml", escape = "none")]
struct BaseTemplate<'a, T: std::fmt::Display> {
    subnets: &'a [T],
    s3_storage_name: &'a str,
}

pub fn get_s3_storage_name(stack_prefix: &str) -> Box<str> {
    format!("{}-s3-storage", stack_prefix).into()
}

pub fn gen_subnet_names(availability_zone_count: usize) -> Box<[Box<str>]> {
    (0..availability_zone_count)
        .map(|i| format!("Subnet{i}").into())
        .collect()
}

pub fn gen_cloudformation_template(
    availability_zone_count: usize,
    stack_prefix: &str,
) -> Box<str> {
    BaseTemplate {
        subnets: &gen_subnet_names(availability_zone_count),
        s3_storage_name: &get_s3_storage_name(stack_prefix),
    }
    .render()
    .expect("Failed to generate template")
    .into()
}

#[test]
fn template_verification_test() {
    let stack = gen_cloudformation_template(3, "trakktor");
    println!("{}", stack);

    assert_eq!(
        crate::hasher::get_hash_value(stack.as_bytes()),
        "1Kax8wzp3tI20SzwA2VvzyPEsAmnSxnzI7NbTyU7e0U"
    )
}
