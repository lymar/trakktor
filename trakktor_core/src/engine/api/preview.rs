use std::cell::RefCell;

use boa_engine::{Context, JsArgs, JsError, JsNativeError, JsResult, JsValue};

use crate::{artifact::Artifact, engine::JsTrkHandle};

pub(super) fn preview(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl Future<Output = JsResult<JsValue>> {
    async move {
        let realm = context.borrow().realm().clone();
        let host_defined = realm.host_defined();
        let Some(th) = host_defined.get::<JsTrkHandle>() else {
            return Err(JsNativeError::typ()
                .with_message("Realm does not have JsTrkHandle field")
                .into());
        };

        const MSG: &str =
            "preview() expects an Artifact object as the first argument";
        let Some(afc) = args.get_or_undefined(0).as_object() else {
            return Err(JsNativeError::typ().with_message(MSG).into());
        };

        let Some(afc) = afc.downcast_ref::<Artifact>() else {
            return Err(JsNativeError::typ().with_message(MSG).into());
        };

        let text = args
            .get(1)
            .map(|v| v.to_string(&mut context.borrow_mut()).ok())
            .flatten()
            .map(|s| s.to_std_string().ok())
            .flatten();

        // log::debug!(afc:?, text:?; "Generating preview for artifact");

        th.trk.preview(&afc, text.as_deref()).await.map_err(|_| {
            <JsNativeError as Into<JsError>>::into(
                JsNativeError::typ().with_message("Failed to render preview"),
            )
        })?;

        Ok(JsValue::undefined())
    }
}
