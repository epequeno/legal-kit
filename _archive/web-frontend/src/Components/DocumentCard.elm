module Components.DocumentCard exposing (view)

import Html exposing (Html, button, div, span, text)
import Html.Attributes exposing (class, style)
import Html.Events exposing (onClick)
import Types exposing (Document)


view : Document -> (String -> msg) -> (String -> msg) -> Html msg
view doc onChat onDelete =
    div
        [ style "background" "white"
        , style "border" "1px solid #e5e7eb"
        , style "border-radius" "8px"
        , style "padding" "20px"
        , style "margin-bottom" "16px"
        , style "box-shadow" "0 1px 3px rgba(0,0,0,0.1)"
        ]
        [ div [ style "display" "flex", style "justify-content" "space-between", style "align-items" "flex-start" ]
            [ div [ style "flex" "1" ]
                [ div
                    [ style "font-size" "18px"
                    , style "font-weight" "600"
                    , style "color" "#111827"
                    , style "margin-bottom" "8px"
                    ]
                    [ text doc.filename ]
                , div
                    [ style "font-size" "14px"
                    , style "color" "#6b7280"
                    , style "margin-bottom" "4px"
                    ]
                    [ text (String.fromInt doc.nChunks ++ " chunks") ]
                , div
                    [ style "font-size" "12px"
                    , style "color" "#9ca3af"
                    ]
                    [ text ("Uploaded: " ++ doc.ingestedAt) ]
                ]
            , div [ style "display" "flex", style "gap" "8px" ]
                [ button
                    [ onClick (onChat doc.id)
                    , style "background" "#2563eb"
                    , style "color" "white"
                    , style "border" "none"
                    , style "border-radius" "6px"
                    , style "padding" "8px 16px"
                    , style "font-size" "14px"
                    , style "font-weight" "500"
                    , style "cursor" "pointer"
                    ]
                    [ text "Chat" ]
                , button
                    [ onClick (onDelete doc.id)
                    , style "background" "#ef4444"
                    , style "color" "white"
                    , style "border" "none"
                    , style "border-radius" "6px"
                    , style "padding" "8px 16px"
                    , style "font-size" "14px"
                    , style "font-weight" "500"
                    , style "cursor" "pointer"
                    ]
                    [ text "Delete" ]
                ]
            ]
        ]
