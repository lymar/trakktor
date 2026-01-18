use std::cell::RefCell;

use boa_engine::{Context, JsArgs, JsError, JsNativeError, JsValue};

pub(crate) fn get_json_arg(
    context: &RefCell<&mut Context>,
    args: &[JsValue],
    idx: usize,
    err_msg: &'static str,
) -> Result<serde_json::Value, JsError> {
    let arg = args
        .get_or_undefined(idx)
        .to_json(&mut context.borrow_mut())?;
    let arg = arg.ok_or_else(|| -> JsError {
        JsNativeError::error().with_message(err_msg).into()
    })?;

    Ok(arg)
}
