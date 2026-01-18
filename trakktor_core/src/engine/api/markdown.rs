use std::borrow::BorrowMut;

use boa_engine::{Context, JsArgs, JsError, JsResult, JsString, JsValue};
use html_to_markdown_rs::convert;

pub(super) fn html_to_markdown(
    _this: &JsValue,
    args: &[JsValue],
    context: &mut Context,
) -> JsResult<JsValue> {
    let html = args
        .get_or_undefined(0)
        .to_string(&mut context.borrow_mut())?
        .to_std_string()
        .map_err(JsError::from_rust)?;

    let markdown = convert(&html, None).map_err(JsError::from_rust)?;

    let res = JsString::from(markdown).into();
    Ok(res)
}
