# LegalKit Frontend - Quick Start Guide

Get the Elm frontend running in 5 minutes.

## Prerequisites

✅ **Node.js 16+** - [Download](https://nodejs.org/)  
✅ **Elm 0.19.1** - Install with `npm install -g elm`  
✅ **Backend running** - FastAPI on `http://localhost:8000`

## Step 1: Install Dependencies

```bash
cd frontend
npm install
```

This installs `elm-land`. Elm packages install automatically on first compile.

## Step 2: Build the Frontend

### Option A: Quick Build Script

```bash
./build.sh
```

### Option B: Manual Build

```bash
elm make src/Main.elm --output=elm.js
```

This will:
1. Download Elm packages (first time only)
2. Type-check all Elm code
3. Compile to `elm.js`

**First build takes ~30 seconds.** Subsequent builds are <5 seconds.

## Step 3: Serve the App

### Option A: Python HTTP Server

```bash
python -m http.server 8080
```

### Option B: npx serve

```bash
npx serve .
```

### Option C: elm-land dev server

```bash
npm start
```

## Step 4: Open in Browser

Navigate to:
```
http://localhost:8080
```

You should see the LegalKit home page with an upload area.

## Step 5: Test the App

### Upload a Document

1. Click the upload area or drag a PDF/TXT file
2. Wait for upload to complete
3. Document appears in the list

### Chat with Document

1. Click "Chat" button on a document
2. Type a question: "What is this document about?"
3. Click "Ask"
4. Watch streaming progress
5. View answer with expandable citations

### Verify Citations

1. Click "▶ Show citations" on a claim
2. See source text quotes
3. Check verification status (green/yellow/red badges)

## Troubleshooting

### "elm: command not found"

Install Elm globally:
```bash
npm install -g elm
```

### Build errors

Check Elm version:
```bash
elm --version  # Should be 0.19.1
```

Clear build cache:
```bash
rm -rf elm-stuff
elm make src/Main.elm --output=elm.js
```

### Blank page

1. Open browser console (F12)
2. Check for JavaScript errors
3. Verify `elm.js` loaded (Network tab)
4. Check that backend is running on port 8000

### Upload fails

1. Check backend is running: `curl http://localhost:8000/api/health`
2. Check browser console for CORS errors
3. Verify file is PDF/TXT/DOCX

### Chat not working

1. Check Network tab for `/ask` request
2. Look for SSE events in console
3. Verify backend SSE format matches expected:
   ```
   event: progress
   data: {...}

   event: result
   data: {...}
   ```

### "Port not found" error

Make sure in `index.html`:
- `main.js` loads **before** `elm.js`
- `setupPorts(app)` is called **after** Elm init

## Development Tips

### Elm Debugger

The Elm compiler will tell you exactly what's wrong. Read error messages carefully!

Example error:
```
-- TYPE MISMATCH ----------------------------------------

The 1st argument to `view` is not what I expect:

10|     view model
              ^^^^^
This `model` value is a:

    Model

But `view` needs the 1st argument to be:

    String
```

### Hot Reloading

For instant feedback:
```bash
# Terminal 1: Auto-rebuild on file change
ls src/**/*.elm | entr elm make src/Main.elm --output=elm.js

# Terminal 2: Serve
python -m http.server 8080
```

### Type-driven Development

1. Change a type
2. Let the compiler tell you what broke
3. Fix each error
4. Everything works when it compiles

### Debugging Ports

Add console.log in `main.js`:
```javascript
app.ports.uploadFile.subscribe((file) => {
  console.log('Uploading:', file.name);
  // ...
});
```

## Production Build

Optimize for production:
```bash
# Build with optimizations
elm make src/Main.elm --output=elm.js --optimize

# Minify (optional but recommended)
npx uglify-js elm.js \
  --compress 'pure_funcs=[F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9],pure_getters,keep_fargs=false,unsafe_comps,unsafe' \
  --output elm.min.js

# Use elm.min.js in index.html
```

**Size comparison:**
- Development: ~200KB
- Optimized: ~100KB
- Minified: ~30KB
- Minified + gzip: ~10KB

## Next Steps

### Customization

- **Colors:** Edit color values in component files
- **API URL:** Change `baseUrl` in `src/Api.elm`
- **Styling:** All styles are inline - search for `style "background"` etc.

### Adding Features

1. Add field to `Types.elm`
2. Update decoder
3. Update UI in relevant component
4. Elm compiler guides you through the rest!

### Learning Elm

- [Official Guide](https://guide.elm-lang.org/)
- [Elm in Action](https://www.manning.com/books/elm-in-action) (book)
- [Exercism Elm Track](https://exercism.org/tracks/elm)

## File Overview

```
index.html       ← Entry point (loads JS)
main.js          ← JavaScript ports (SSE, uploads)
elm.js           ← Compiled Elm (generated)
src/Main.elm     ← App entry + routing
src/Types.elm    ← Domain model + decoders
src/Api.elm      ← HTTP calls
src/Page/*.elm   ← Page components
src/Components/*.elm ← Reusable UI components
```

## Common Tasks

### Change API URL

```elm
-- src/Api.elm
baseUrl : String
baseUrl =
    "https://api.myapp.com"  -- Change this
```

### Add a new route

```elm
-- src/Main.elm
type Route
    = Home
    | ChatPage String
    | NewPage  -- Add this

routeParser : Parser (Route -> a) a
routeParser =
    Parser.oneOf
        [ Parser.map Home Parser.top
        , Parser.map ChatPage (Parser.s "documents" </> Parser.string)
        , Parser.map NewPage (Parser.s "new")  -- Add this
        ]
```

### Style changes

All styles are inline. Example:
```elm
-- Search for the element, then edit style attributes
div
    [ style "background" "#ff0000"  -- Change color
    , style "padding" "20px"         -- Change padding
    ]
    [ text "Hello" ]
```

## Support

- **Elm Slack:** [Join](https://elmlang.herokuapp.com/)
- **Discourse:** [Elm Discourse](https://discourse.elm-lang.org/)
- **GitHub Issues:** Report bugs in the LegalKit repo

## Success Checklist

- [ ] Elm installed and version is 0.19.1
- [ ] npm install completed
- [ ] `./build.sh` completed without errors
- [ ] Server running on port 8080
- [ ] Backend running on port 8000
- [ ] Page loads in browser
- [ ] No console errors
- [ ] Can upload a document
- [ ] Can navigate to chat
- [ ] Can ask a question
- [ ] Can see streaming response
- [ ] Citations expand/collapse

**All checked?** You're ready to go! 🎉
