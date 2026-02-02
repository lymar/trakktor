use std::cell::RefCell;

use boa_engine::{
    Context, JsArgs, JsError, JsNativeError, JsObject, JsResult, JsValue,
};

use crate::engine::JsTrkHandle;

pub(super) fn read_file(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl Future<Output = JsResult<JsValue>> {
    async move {
        let path = args
            .get_or_undefined(0)
            .to_string(&mut context.borrow_mut())?
            .to_std_string()
            .map_err(JsError::from_rust)?;

        let realm = context.borrow().realm().clone();
        let host_defined = realm.host_defined();
        let Some(trkh) = host_defined.get::<JsTrkHandle>() else {
            return Err(JsNativeError::typ()
                .with_message("Realm does not have JsTrkHandle field")
                .into());
        };

        let afc = trkh.trk.read_file(&path).await.map_err(|_| -> JsError {
            JsNativeError::error()
                .with_message("Failed to read file")
                .into()
        })?;

        log::debug!("Reading file at path: {}", path);

        Ok(JsObject::from_proto_and_data(None, afc).into())
    }
}
