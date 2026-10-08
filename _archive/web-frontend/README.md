# LegalKit Frontend

Clean, minimal Elm frontend for the LegalKit MVP.

## Features

- **Home Page** (`/`) - Upload documents and view document list
- **Chat Page** (`/documents/:id`) - Ask questions with streaming citations

## Tech Stack

- **Elm 0.19.1** - Type-safe functional frontend
- **elm-css** - CSS-in-Elm styling
- **JavaScript Ports** - SSE streaming and file uploads

## Setup

### Prerequisites

- Node.js 16+
- Elm 0.19.1

### Install Dependencies

```bash
npm install
```

This will install `elm-land` as a dev dependency. Elm packages will be installed automatically when you first compile.

## Development

### Option 1: Elm Reactor (Simple)

```bash
# Compile Elm code
elm make src/Main.elm --output=elm.js

# Serve with a simple HTTP server
python -m http.server 8080
# or
npx serve .
```

Then open http://localhost:8080

### Option 2: Elm Land (Recommended)

```bash
npm start
```

This will start the elm-land development server with hot reloading.

## Build for Production

```bash
# Build optimized Elm
elm make src/Main.elm --output=elm.js --optimize

# Minify (optional)
npx uglify-js elm.js --compress 'pure_funcs=[F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9],pure_getters,keep_fargs=false,unsafe_comps,unsafe' --output elm.min.js

# Serve static files
# Copy index.html, main.js, and elm.js (or elm.min.js) to your web server
```

## API Configuration

The frontend expects the FastAPI backend to be running on `http://localhost:8000`.

To change this, edit the `baseUrl` in `src/Api.elm`.

## File Structure

```
frontend/
├── elm.json                    # Elm dependencies
├── package.json                # npm dependencies
├── elm-land.json              # elm-land config
├── index.html                  # Entry point
├── main.js                     # JavaScript ports (SSE, uploads)
└── src/
    ├── Main.elm               # App entry + routing
    ├── Api.elm                # HTTP API calls
    ├── Types.elm              # Shared types + decoders
    ├── Page/
    │   ├── Home.elm           # Home page
    │   └── Chat.elm           # Chat page
    └── Components/
        ├── DocumentCard.elm    # Document list item
        ├── CitationCard.elm    # Claim + citations display
        └── UploadArea.elm      # File upload component
```

## How It Works

### Routing

- Uses `Browser.application` for URL-based routing
- Routes:
  - `/` → Home page
  - `/documents/:id` → Chat page

### File Uploads

1. User selects file in `UploadArea`
2. Elm sends `File` through `uploadFile` port
3. JavaScript creates FormData and POSTs to `/api/documents`
4. Result sent back through `uploadComplete` port
5. Document added to list

### Streaming Chat

1. User submits question
2. Elm sends request through `startAsk` port
3. JavaScript opens SSE stream (POST to `/api/documents/:id/ask`)
4. Progress events shown in real-time
5. Final result displayed with expandable citations

## Port Interface

### Elm → JavaScript

- `uploadFile : File -> Cmd msg` - Upload document
- `startAsk : {docId, question, topK} -> Cmd msg` - Start streaming ask

### JavaScript → Elm

- `uploadComplete : {ok, doc?, error?} -> Sub msg` - Upload result
- `askEvent : {event, message|data} -> Sub msg` - SSE events
  - `progress` - Stage updates
  - `result` - Final answer with citations
  - `error` - Error message

## Styling

- Clean, minimal design with system font stack
- Colors:
  - Background: `#f8f9fa`
  - Primary: `#2563eb` (blue)
  - Verified: `#16a34a` (green)
  - Partial: `#d97706` (amber)
  - Unverified: `#dc2626` (red)
- Max content width: 960px
- Responsive layout with inline styles

## Troubleshooting

### Elm compilation errors

Make sure you have Elm 0.19.1 installed:
```bash
elm --version
```

### Port not found errors

Make sure `main.js` is loaded before `elm.js` in `index.html`.

### CORS issues

The backend must allow CORS from your frontend origin. Check that the FastAPI backend has proper CORS middleware configured.

### SSE streaming not working

Check browser console for errors. Make sure the backend is returning proper SSE format:
```
event: progress
data: {"stage": "..."}

event: result
data: {"answer": "...", ...}

```
