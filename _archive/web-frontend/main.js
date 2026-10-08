// main.js - JavaScript interop for Elm ports

function setupPorts(app) {
  // Port: Upload file
  if (app.ports.uploadFile) {
    app.ports.uploadFile.subscribe(async (file) => {
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('http://localhost:8000/api/documents', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const doc = await response.json();
        app.ports.uploadComplete.send({ ok: true, doc: doc });
      } catch (error) {
        app.ports.uploadComplete.send({ 
          ok: false, 
          error: error.message 
        });
      }
    });
  }

  // Port: Start SSE streaming ask
  if (app.ports.startAsk) {
    app.ports.startAsk.subscribe(async ({ docId, question, topK }) => {
      try {
        const response = await fetch(`http://localhost:8000/api/documents/${docId}/ask`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: question,
            top_k: topK
          })
        });

        if (!response.ok) {
          throw new Error(`Request failed: ${response.statusText}`);
        }

        // Read the SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          
          // Parse SSE events from buffer
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          let currentEvent = { event: null, data: '' };

          for (const line of lines) {
            if (line.startsWith('event:')) {
              currentEvent.event = line.substring(6).trim();
            } else if (line.startsWith('data:')) {
              currentEvent.data = line.substring(5).trim();
            } else if (line === '') {
              // Empty line marks end of event
              if (currentEvent.event) {
                handleSSEEvent(app, currentEvent);
                currentEvent = { event: null, data: '' };
              }
            }
          }
        }
      } catch (error) {
        app.ports.askEvent.send({
          event: 'error',
          message: error.message
        });
      }
    });
  }
}

function handleSSEEvent(app, sseEvent) {
  try {
    if (sseEvent.event === 'progress') {
      const data = JSON.parse(sseEvent.data);
      app.ports.askEvent.send({
        event: 'progress',
        message: data.message || `${data.stage}...`
      });
    } else if (sseEvent.event === 'result') {
      const data = JSON.parse(sseEvent.data);
      app.ports.askEvent.send({
        event: 'result',
        data: data
      });
    } else if (sseEvent.event === 'error') {
      const data = JSON.parse(sseEvent.data);
      app.ports.askEvent.send({
        event: 'error',
        message: data.message || 'An error occurred'
      });
    }
  } catch (error) {
    console.error('Error parsing SSE event:', error, sseEvent);
    app.ports.askEvent.send({
      event: 'error',
      message: 'Failed to parse server response'
    });
  }
}

// Export for use in index.html
if (typeof window !== 'undefined') {
  window.setupPorts = setupPorts;
}
