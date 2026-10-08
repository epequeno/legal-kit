module Components.UploadArea exposing (view)

import File exposing (File)
import Html exposing (Html, div, input, label, text)
import Html.Attributes exposing (accept, id, style, type_)
import Html.Events exposing (on)
import Json.Decode as Decode


view : (File -> msg) -> Maybe String -> Html msg
view onFileSelected uploadingMsg =
    div
        [ style "background" "white"
        , style "border" "2px dashed #cbd5e1"
        , style "border-radius" "8px"
        , style "padding" "40px"
        , style "text-align" "center"
        , style "margin-bottom" "32px"
        , style "cursor" "pointer"
        ]
        [ label
            [ style "cursor" "pointer"
            , style "display" "block"
            ]
            [ input
                [ type_ "file"
                , id "file-upload"
                , accept ".pdf,.txt,.docx"
                , style "display" "none"
                , on "change" (Decode.map onFileSelected fileDecoder)
                ]
                []
            , div []
                [ div
                    [ style "font-size" "48px"
                    , style "margin-bottom" "16px"
                    ]
                    [ text "📄" ]
                , div
                    [ style "font-size" "18px"
                    , style "font-weight" "600"
                    , style "color" "#374151"
                    , style "margin-bottom" "8px"
                    ]
                    [ text "Upload a document" ]
                , div
                    [ style "font-size" "14px"
                    , style "color" "#6b7280"
                    ]
                    [ text "Click to select or drag and drop" ]
                , div
                    [ style "font-size" "12px"
                    , style "color" "#9ca3af"
                    , style "margin-top" "4px"
                    ]
                    [ text "PDF, TXT, or DOCX" ]
                , case uploadingMsg of
                    Just msg ->
                        div
                            [ style "margin-top" "16px"
                            , style "font-size" "14px"
                            , style "color" "#2563eb"
                            , style "font-weight" "500"
                            ]
                            [ text msg ]

                    Nothing ->
                        text ""
                ]
            ]
        ]


fileDecoder : Decode.Decoder File
fileDecoder =
    Decode.at [ "target", "files", "0" ] File.decoder
