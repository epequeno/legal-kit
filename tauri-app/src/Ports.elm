port module Ports exposing
    -- Outbound (Elm → JS → Tauri)
    ( copyToClipboard
    , invokeGetDocumentOutline
    , invokeIngestDocument
    , invokeAskQuestion
    , invokeListDocuments
    , invokeDeleteDocument
    , invokeGetApiKeyStatus
    , invokeSetApiKeys
    , invokeLoadChatHistory
    , invokeClearChatHistory
    , invokeSaveSettings
    , invokeGetSettings
    , invokeClearAllData
    , openFilePicker
    -- Inbound (Tauri → JS → Elm)
    , onIngestResult
    , onAskResult
    , onListResult
    , onDeleteResult
    , onApiKeyStatus
    , onFileSelected
    , onChatHistoryResult
    , onSettingsResult
    , onSaveSettingsDone
    , onClearAllDone
    , onDocumentOutline
    , onTauriError
    )

import Json.Encode as Encode


-- ── Outbound ─────────────────────────────────────────────────────────────


port copyToClipboard : String -> Cmd msg

port invokeGetDocumentOutline : String -> Cmd msg

port invokeIngestDocument : String -> Cmd msg

port invokeAskQuestion : Encode.Value -> Cmd msg

port invokeListDocuments : () -> Cmd msg

port invokeDeleteDocument : String -> Cmd msg

port invokeGetApiKeyStatus : () -> Cmd msg

port invokeSetApiKeys : Encode.Value -> Cmd msg

port openFilePicker : () -> Cmd msg

port invokeLoadChatHistory : String -> Cmd msg

port invokeClearChatHistory : String -> Cmd msg

port invokeSaveSettings : Encode.Value -> Cmd msg

port invokeGetSettings : () -> Cmd msg

port invokeClearAllData : () -> Cmd msg


-- ── Inbound ──────────────────────────────────────────────────────────────


port onIngestResult : (Encode.Value -> msg) -> Sub msg

port onAskResult : (Encode.Value -> msg) -> Sub msg

port onListResult : (Encode.Value -> msg) -> Sub msg

port onDeleteResult : (Encode.Value -> msg) -> Sub msg

port onApiKeyStatus : (Encode.Value -> msg) -> Sub msg

port onFileSelected : (String -> msg) -> Sub msg

port onTauriError : (String -> msg) -> Sub msg

port onChatHistoryResult : (Encode.Value -> msg) -> Sub msg

port onSettingsResult : (Encode.Value -> msg) -> Sub msg

port onSaveSettingsDone : (() -> msg) -> Sub msg

port onClearAllDone : (() -> msg) -> Sub msg

port onDocumentOutline : (Encode.Value -> msg) -> Sub msg
