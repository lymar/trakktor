use std::cell::RefCell;

use boa_engine::{Context, JsArgs, JsError, JsResult, JsString, JsValue};

pub(super) fn http_get_text(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl Future<Output = JsResult<JsValue>> {
    async move {
        let url = args
            .get_or_undefined(0)
            .to_string(&mut context.borrow_mut())?
            .to_std_string()
            .map_err(JsError::from_rust)?;

        let client = reqwest::Client::builder()
            .user_agent("Trakktor")
            .build()
            .map_err(JsError::from_rust)?;

        let resp = client
            .get(url)
            .send()
            .await
            .map_err(JsError::from_rust)?
            .text()
            .await
            .map_err(JsError::from_rust)?;

        let res = JsString::from(resp).into();
        Ok(res)
    }
}
