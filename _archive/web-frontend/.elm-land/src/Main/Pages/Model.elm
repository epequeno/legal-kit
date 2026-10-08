module Main.Pages.Model exposing (Model(..))

import Pages.NotFound_
import View exposing (View)


type Model
    = NotFound_ Pages.NotFound_.Model
    | Redirecting_
    | Loading_
