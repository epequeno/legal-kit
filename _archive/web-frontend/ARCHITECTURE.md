# LegalKit Frontend Architecture

## Overview

Clean, minimal Elm frontend using The Elm Architecture (TEA) with JavaScript ports for SSE streaming and file uploads.

## File Structure

```
frontend/
├── elm.json                    # Elm dependencies (elm/browser, elm/http, rtfeldman/elm-css)
├── package.json                # npm dependencies (elm-land)
├── elm-land.json              # elm-land configuration
├── index.html                  # HTML entry point
├── main.js                     # JavaScript ports implementation
├── build.sh                    # Build script
├── README.md                   # Documentation
├── ARCHITECTURE.md            # This file
└── src/
    ├── Main.elm               # App entry, routing, top-level orchestration
    ├── Api.elm                # HTTP API calls (non-streaming)
    ├── Types.elm              # Shared domain types + JSON decoders
    ├── Page/
    │   ├── Home.elm           # Home page: upload + document list
    │   └── Chat.elm           # Chat page: streaming Q&A interface
    └── Components/
        ├── DocumentCard.elm    # Single document in list
        ├── CitationCard.elm    # Claim + citations display
        └── UploadArea.elm      # Drag-drop upload zone
```

## Architecture Layers

### 1. Main.elm - Application Shell

**Responsibilities:**
- URL routing using `Browser.application`
- Page initialization based on route
- Message delegation to pages
- Navigation coordination

**Routes:**
- `/` → Home page
- `/documents/:id` → Chat page
- `*` → 404

**Key Types:**
```elm
type Route = Home | ChatPage String | NotFound
type Page = HomePage Home.Model | ChatPageModel Chat.Model | NotFoundPage
```

### 2. Types.elm - Domain Model

**Core Types:**

```elm
Document     -- id, filename, sizeBytes, nChunks, ingestedAt
Citation     -- sectionId, sourceText, status, entailment, score
Claim        -- text, isGap, hasHedging, confidence, citations
Confidence   -- level (high/medium/low), score
AskResult    -- answer, claims, overallConfidence, gaps, elapsed
```

**JSON Decoders:**
- All types have corresponding decoders for API responses
- Snake_case API fields mapped to camelCase Elm fields

### 3. Api.elm - HTTP Layer

**Endpoints:**
```elm
getHealth       : GET  /api/health
getDocuments    : GET  /api/documents
getDocument     : GET  /api/documents/:id
deleteDocument  : DELETE /api/documents/:id
```

**Note:** Streaming `/ask` endpoint uses JavaScript ports, not Http module.

### 4. Page Modules

#### Page/Home.elm

**Features:**
- Load and display document list on init
- File upload with progress indicator
- Delete documents
- Navigate to chat page

**State:**
```elm
{ documents : List Document
, loading : Bool
, error : Maybe String
, uploading : Maybe String  -- progress message
}
```

**Ports:**
- `uploadFile` : Send File to JS
- `uploadComplete` : Receive upload result

**Messages:**
- `GotDocuments` - API response
- `FileSelected` - User picked file
- `UploadComplete` - Upload finished
- `DeleteDocument` - User clicked delete
- `DocumentDeleted` - Delete response
- `NavigateToChat` - Navigate to chat (handled by Main)

#### Page/Chat.elm

**Features:**
- Display document header
- Message history (user questions + assistant answers)
- Real-time streaming progress
- Expandable citation panels
- Knowledge gap warnings

**State:**
```elm
{ docId : String
, document : Maybe Document
, messages : List Message
, questionInput : String
, loading : Bool
, streamingProgress : Maybe String
, expandedClaims : Dict Int Bool  -- track which claims are expanded
, error : Maybe String
}

type Message
  = UserMessage String
  | AssistantMessage AskResult
```

**Ports:**
- `startAsk` : Send {docId, question, topK} to JS
- `askEvent` : Receive SSE events

**SSE Event Types:**
```elm
type AskEvent
  = ProgressEvent String       -- "Finding passages...", "Generating..."
  | ResultEvent AskResult       -- Final answer with claims
  | ErrorEvent String           -- Error message
```

**Messages:**
- `GotDocument` - Loaded doc metadata
- `QuestionInputChanged` - User typing
- `SubmitQuestion` - User submitted
- `AskEvent` - SSE event received
- `ToggleClaim` - Expand/collapse citations
- `NavigateBack` - Go to home (handled by Main)

### 5. Component Modules

#### Components/DocumentCard.elm

**Props:**
- `Document` - doc to display
- `onChat : String -> msg` - chat button handler
- `onDelete : String -> msg` - delete button handler

**Renders:**
- Filename (title)
- Chunk count
- Upload timestamp
- "Chat" button (primary blue)
- "Delete" button (red)

#### Components/CitationCard.elm

**Props:**
- `Claim` - claim to display
- `Bool` - expanded state
- `msg` - toggle expansion message

**Renders:**
- Claim text
- Status badge (verified/partial/unverified)
- Confidence badge
- Gap/hedging badges (if applicable)
- Expandable citation list
  - Section ID
  - Source text quote
  - Entailment score

**Status Logic:**
- All citations verified → Green "Verified"
- Some verified → Amber "Partial"
- None verified → Red "Unverified"
- No citations → Gray "No citations"

#### Components/UploadArea.elm

**Props:**
- `onFileSelected : File -> msg` - file picked handler
- `Maybe String` - upload progress message

**Features:**
- Click to select file
- Shows upload progress when active
- Accepts .pdf, .txt, .docx

**Note:** Drag-and-drop requires additional JS event handlers (not implemented yet).

### 6. JavaScript Interop (main.js)

**Port Implementations:**

#### uploadFile Port
1. Elm sends `File` object
2. JS creates `FormData`
3. POST to `/api/documents`
4. Parse response JSON
5. Send back `{ok, doc}` or `{ok, error}`

#### startAsk Port
1. Elm sends `{docId, question, topK}`
2. JS POSTs to `/api/documents/:id/ask`
3. Reads SSE stream using `ReadableStream` API
4. Parses SSE format (event:/data: lines)
5. Sends each event to Elm via `askEvent` port

**SSE Parsing:**
```
event: progress
data: {"stage": "Finding passages", "message": "..."}

event: result
data: {"answer": "...", "claims": [...], ...}

event: error
data: {"message": "..."}
```

## Data Flow Examples

### Upload Flow

```
User selects file
  → UploadArea.view captures File
  → Home.update FileSelected
  → Port: uploadFile
  → JS: FormData POST
  → API: /api/documents
  ← JS: uploadComplete port
  ← Home.update UploadComplete
  ← Add doc to list
```

### Ask Flow

```
User submits question
  → Chat.update SubmitQuestion
  → Add UserMessage to messages
  → Port: startAsk
  → JS: SSE POST /ask
  
  [Stream events]
  ← Port: askEvent (progress)
  ← Chat.update: show progress
  
  ← Port: askEvent (result)
  ← Chat.update: add AssistantMessage
  ← Render answer + citations
```

### Navigation Flow

```
User clicks "Chat" button
  → DocumentCard onClick
  → Home.update NavigateToChat
  → Main.update: Nav.pushUrl
  → Browser URL changes
  → Main.update UrlChanged
  → Parse URL → ChatPage route
  → Init Chat.Model
  → Chat.init: fetch document
  → Render chat page
```

## Styling Approach

**Method:** Inline styles (not elm-css in the end)

- No separate CSS files
- All styles in Elm via `Html.Attributes.style`
- Consistent color palette
- System font stack
- Flexbox-based layout

**Colors:**
```elm
Background:    #f8f9fa
White:         #ffffff
Primary:       #2563eb
Verified:      #16a34a
Partial:       #d97706
Unverified:    #dc2626
Text Primary:  #111827
Text Secondary:#6b7280
Border:        #e5e7eb
```

## Error Handling

**HTTP Errors:**
- Displayed in red error banner
- Stored in `error : Maybe String` field
- User can retry by re-triggering action

**SSE Errors:**
- Sent as `error` event from backend
- Displayed in error banner
- Streaming stopped

**Upload Errors:**
- Network errors caught in JS
- Sent back through `uploadComplete` port
- Displayed in error banner

## Type Safety Guarantees

1. **Routing:** All routes type-checked at compile time
2. **JSON:** All API responses decoded with explicit decoders (runtime safety)
3. **Messages:** Impossible states ruled out (e.g., can't have AssistantMessage without AskResult)
4. **Navigation:** Can't navigate to invalid routes

## Future Enhancements

### Easy Additions
- Drag-and-drop file upload (add JS event listeners)
- Loading spinners (replace text with animated elements)
- Markdown rendering in answers
- Export chat history

### Moderate Complexity
- Multi-document chat
- Search/filter documents
- Pagination for large document lists
- Real-time document processing progress

### Major Features
- Authentication
- Saved chat sessions
- Document annotations
- Side-by-side document viewer

## Performance Considerations

**Current:**
- No virtualization (fine for <100 documents)
- No lazy loading
- All messages in memory

**Future Optimizations:**
- Virtual scrolling for long message lists
- Pagination for document list
- Lazy image loading for citations
- Debounced search input

## Testing Strategy

**Unit Tests:**
- JSON decoders (feed known JSON, verify parsing)
- Route parsing (URL → Route)

**Integration Tests:**
- Port communication (mock JS responses)
- Page navigation flow

**E2E Tests:**
- Upload document
- Ask question
- View citations
- Delete document

## Development Workflow

1. **Install:** `npm install`
2. **Build:** `./build.sh` or `elm make src/Main.elm --output=elm.js`
3. **Serve:** `python -m http.server 8080`
4. **Test:** Open browser, check console for errors

**Debugging:**
- Use Elm debugger (time-travel debugging)
- Browser console for JS port errors
- Network tab for API requests
- Elm compiler errors are extremely helpful

## Deployment

**Static Hosting:**
1. Build optimized: `elm make src/Main.elm --optimize --output=elm.js`
2. Minify: `uglify-js elm.js --compress ... --output=elm.min.js`
3. Copy files: `index.html`, `main.js`, `elm.min.js`
4. Serve static files

**Backend Integration:**
- Update `baseUrl` in `Api.elm` to production API
- Ensure CORS configured correctly
- Serve from same domain to avoid CORS entirely

**CDN:**
- Upload static assets to CDN
- Update script src paths
- Set cache headers

## Dependencies

**Elm Packages:**
- `elm/browser` - Browser.application
- `elm/core` - Core language
- `elm/html` - HTML generation
- `elm/http` - HTTP requests
- `elm/json` - JSON encoding/decoding
- `elm/url` - URL parsing
- `elm/file` - File handling
- `rtfeldman/elm-css` - CSS-in-Elm (installed but not used yet)

**JavaScript:**
- Vanilla JS (no framework)
- Fetch API for HTTP
- ReadableStream API for SSE
- FormData API for uploads

**npm:**
- `elm-land` - Development server and tooling (optional)

## API Contract

The frontend expects the backend to conform to this API:

```
GET  /api/health
  → 200 "ok"

GET  /api/documents
  → 200 [{id, filename, size_bytes, n_chunks, ingested_at}]

POST /api/documents (multipart/form-data)
  file: <file>
  → 200 {id, filename, size_bytes, n_chunks, ingested_at}

GET  /api/documents/:id
  → 200 {id, filename, size_bytes, n_chunks, ingested_at}

DELETE /api/documents/:id
  → 204

POST /api/documents/:id/ask
  {question: string, top_k: int}
  → SSE stream:
    event: progress
    data: {stage: string, message: string}
    
    event: result
    data: {answer, claims, overall_confidence, gaps, elapsed_seconds, doc_id}
    
    event: error
    data: {message: string}
```

All field names in snake_case (API) → camelCase (Elm).
