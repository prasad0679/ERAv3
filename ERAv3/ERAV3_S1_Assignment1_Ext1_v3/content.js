(function() {
  let summarizerActive = false;
  let currentUtterance = null;

  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Message received:", request);
    if (request.action === "openFloatingWindow") {
      openFloatingWindow();
    }
  });

  function initializeSummarizer() {
    console.log("Initializing summarizer");
    const floatingWindow = document.getElementById('summarizer-floating-window');
    const summarizerButton = document.getElementById('summarizer-button');
    const wordCountSelect = document.getElementById('summarizer-word-count');
    const summarizerClose = document.getElementById('summarizer-close');
    const summarizerReset = document.getElementById('summarizer-reset');
    const apiKeyInput = document.getElementById('openai-api-key');

    makeDraggable(floatingWindow);

    summarizerButton.addEventListener('click', handleSummarizerClick);
    summarizerClose.addEventListener('click', closeFloatingWindow);
    summarizerReset.addEventListener('click', resetFloatingWindow);
    
    // Ensure the dropdown and input are clickable and not affected by the draggable behavior
    wordCountSelect.addEventListener('mousedown', (e) => {
      e.stopPropagation();
    });
    apiKeyInput.addEventListener('mousedown', (e) => {
      e.stopPropagation();
    });

    wordCountSelect.addEventListener('change', updateDefaultWordCount);
    apiKeyInput.addEventListener('change', saveApiKey);

    // Always set the default word count
    const defaultWordCount = '100';
    wordCountSelect.value = defaultWordCount;
    localStorage.setItem('summarizerDefaultWordCount', defaultWordCount);

    // Always clear the API key input
    apiKeyInput.value = '';

    // Debug log
    console.log('Initializing summarizer with word count:', defaultWordCount);
  }

  function getSelectedText() {
    return window.getSelection().toString().trim();
  }

  function handleSummarizerClick() {
    console.log("Summarizer button clicked");
    const selectedText = getSelectedText();
    if (!selectedText) {
      alert('Please select some text on the page to summarize.');
      return;
    }

    const apiKey = document.getElementById('openai-api-key').value;
    if (!apiKey) {
      alert('Please enter your OpenAI API key.');
      return;
    }

    const wordCount = document.getElementById('summarizer-word-count').value;

    const summarizerButton = document.getElementById('summarizer-button');
    if (summarizerButton.disabled) {
      return; // Prevent multiple clicks while processing
    }

    summarizerButton.disabled = true;
    summarizerButton.textContent = 'Summarizing...';

    console.log("Calling summarizeText function");
    summarizeText(selectedText, wordCount, apiKey)
      .then(summary => {
        console.log("Summary received, showing summary window");
        showSummaryWindow(summary);
      })
      .catch(error => {
        console.error('Error:', error);
        alert(`An error occurred while summarizing the text: ${error.message}`);
      })
      .finally(() => {
        summarizerButton.disabled = false;
        summarizerButton.textContent = 'S';
      });
  }

  function showMessage(message) {
    const messagePopup = document.createElement('div');
    messagePopup.id = 'summarizer-message';
    messagePopup.innerHTML = `
      <p>${message}</p>
      <button id="summarizer-message-close">Close</button>
    `;
    document.body.appendChild(messagePopup);

    document.getElementById('summarizer-message-close').addEventListener('click', () => {
      messagePopup.remove();
    });
  }

  function updateDefaultWordCount(event) {
    const wordCountSelect = event.target;
    localStorage.setItem('summarizerDefaultWordCount', wordCountSelect.value);
    console.log('Updated default word count:', wordCountSelect.value); // Debug log
  }

  function showSummaryWindow(summary) {
    console.log("Creating summary window");
    const summaryWindow = document.createElement('div');
    summaryWindow.id = 'summarizer-summary-window';
    summaryWindow.innerHTML = `
      <div class="draggable-handle">Summary</div>
      <div class="summary-content">
        <p>${summary}</p>
      </div>
      <button id="summarizer-summary-close">Close</button>
      <button id="summarizer-read-aloud">Read Aloud</button>
      <button id="summarizer-export">Export</button>
      <div class="resize-handle"></div>
    `;
    
    // Add inline styles to ensure visibility and resizability
    summaryWindow.style.cssText = `
      position: fixed;
      top: 50px;
      left: 50px;
      width: 300px;
      height: 400px;
      background-color: white;
      border: 1px solid black;
      padding: 10px;
      z-index: 9999;
      overflow-y: auto;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
      resize: both;
      overflow: auto;
    `;
    
    document.body.appendChild(summaryWindow);

    console.log("Summary window created and appended to body");

    makeDraggable(summaryWindow);
    makeResizable(summaryWindow);

    document.getElementById('summarizer-summary-close').addEventListener('click', closeSummaryWindow);
    document.getElementById('summarizer-read-aloud').addEventListener('click', () => readAloud(summary));
    document.getElementById('summarizer-export').addEventListener('click', () => exportSummary(summary));

    console.log("Event listeners added to summary window buttons");
  }

  function closeSummaryWindow() {
    const summaryWindow = document.getElementById('summarizer-summary-window');
    if (summaryWindow) {
      summaryWindow.remove();
    }
    stopReadAloud();
    summarizerActive = false;
    
    // Clear the OpenAI API key from the floating window
    const apiKeyInput = document.getElementById('openai-api-key');
    if (apiKeyInput) {
      apiKeyInput.value = '';
    }
  }

  function makeDraggable(element) {
    console.log("Making element draggable:", element);
    const handle = element.querySelector('.draggable-handle');
    if (!handle) {
      console.error("Draggable handle not found");
      return;
    }

    let isDragging = false;
    let startX, startY, startLeft, startTop;

    handle.addEventListener('mousedown', startDragging);

    function startDragging(e) {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      startLeft = parseInt(window.getComputedStyle(element).left);
      startTop = parseInt(window.getComputedStyle(element).top);
      document.addEventListener('mousemove', drag);
      document.addEventListener('mouseup', stopDragging);
    }

    function drag(e) {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      element.style.left = `${startLeft + dx}px`;
      element.style.top = `${startTop + dy}px`;
    }

    function stopDragging() {
      isDragging = false;
      document.removeEventListener('mousemove', drag);
      document.removeEventListener('mouseup', stopDragging);
    }
  }

  function makeResizable(element) {
    const resizeHandle = element.querySelector('.resize-handle');
    let isResizing = false;
    let originalWidth, originalHeight, originalX, originalY;

    resizeHandle.addEventListener('mousedown', startResize);

    function startResize(e) {
      isResizing = true;
      originalWidth = parseFloat(getComputedStyle(element, null).getPropertyValue('width').replace('px', ''));
      originalHeight = parseFloat(getComputedStyle(element, null).getPropertyValue('height').replace('px', ''));
      originalX = e.pageX;
      originalY = e.pageY;
      document.addEventListener('mousemove', resize);
      document.addEventListener('mouseup', stopResize);
      e.preventDefault();
    }

    function resize(e) {
      if (!isResizing) return;
      const width = originalWidth + (e.pageX - originalX);
      const height = originalHeight + (e.pageY - originalY);
      element.style.width = width + 'px';
      element.style.height = height + 'px';
    }

    function stopResize() {
      isResizing = false;
      document.removeEventListener('mousemove', resize);
      document.removeEventListener('mouseup', stopResize);
    }
  }

  function exportSummary(summary) {
    // Copy to clipboard
    navigator.clipboard.writeText(summary).then(() => {
      alert('Summary copied to clipboard. You can now paste it into OneNote or Notepad.');
      // Removed the closeFloatingWindow() call
    }).catch(err => {
      console.error('Failed to copy text: ', err);
      alert('Failed to copy summary. Please try again.');
    });
  }

  function readAloud(text) {
    stopReadAloud();
    currentUtterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(currentUtterance);
  }

  function stopReadAloud() {
    if (currentUtterance) {
      window.speechSynthesis.cancel();
      currentUtterance = null;
    }
  }

  async function summarizeText(text, length, apiKey) {
    const apiUrl = 'https://api.openai.com/v1/chat/completions';
    
    const prompt = `Summarize the following text in approximately ${length} words:\n\n${text}`;

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are a helpful assistant that summarizes text.' },
            { role: 'user', content: prompt }
          ],
          max_tokens: parseInt(length) * 2,
          n: 1,
          stop: null,
          temperature: 0.7,
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error?.message || response.statusText;
        } catch (e) {
          errorMessage = errorText || response.statusText;
        }
        throw new Error(`API error: ${errorMessage}`);
      }

      const data = await response.json();
      return data.choices[0].message.content.trim();
    } catch (error) {
      console.error('Error in summarizeText:', error);
      throw error;
    }
  }

  function showLoadingIndicator() {
    const loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'summarizer-loading';
    loadingIndicator.textContent = 'Generating summary...';
    document.body.appendChild(loadingIndicator);
  }

  function hideLoadingIndicator() {
    const loadingIndicator = document.getElementById('summarizer-loading');
    if (loadingIndicator) {
      loadingIndicator.remove();
    }
  }

  function openFloatingWindow() {
    // Remove existing floating window if it exists
    const existingWindow = document.getElementById('summarizer-floating-window');
    if (existingWindow) {
      existingWindow.remove();
    }

    // Create a new floating window
    const floatingWindow = document.createElement('div');
    floatingWindow.id = 'summarizer-floating-window';
    floatingWindow.innerHTML = `
      <div id="summarizer-handle" class="draggable-handle">
        <input type="password" id="openai-api-key" placeholder="Enter OpenAI API key">
        <button id="summarizer-button">S</button>
        <div class="word-count-container">
          <select id="summarizer-word-count">
            ${generateWordCountOptions()}
          </select>
          <span>words</span>
        </div>
        <button id="summarizer-reset">Reset</button>
        <button id="summarizer-close">C</button>
      </div>
    `;
    document.body.appendChild(floatingWindow);
    initializeSummarizer();

    // Adjust floating window size and style
    const style = document.createElement('style');
    style.textContent = `
      #summarizer-floating-window {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        z-index: 9999;
        width: 350px;
      }
      #summarizer-handle {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 5px;
      }
      #openai-api-key {
        flex: 1 1 100%;
        margin-bottom: 5px;
      }
      #summarizer-button, #summarizer-close, #summarizer-reset {
        flex: 0 0 auto;
      }
      .word-count-container {
        flex: 1 1 auto;
        display: flex;
        align-items: center;
        gap: 5px;
      }
      #summarizer-word-count {
        flex: 1 1 auto;
      }
    `;
    document.head.appendChild(style);
  }

  function generateWordCountOptions() {
    let options = '';
    const defaultWordCount = localStorage.getItem('summarizerDefaultWordCount') || '100';
    for (let i = 50; i <= 500; i += 50) {
      options += `<option value="${i}" ${i.toString() === defaultWordCount ? 'selected' : ''}>${i}</option>`;
    }
    return options;
  }

  function closeFloatingWindow() {
    const floatingWindow = document.getElementById('summarizer-floating-window');
    if (floatingWindow) {
      const apiKeyInput = document.getElementById('openai-api-key');
      if (apiKeyInput) {
        apiKeyInput.value = '';
      }
      floatingWindow.remove(); // This removes the floating window from the DOM
    }
    closeSummaryWindow();
  }

  function saveApiKey() {
    const apiKey = document.getElementById('openai-api-key').value;
    chrome.storage.sync.set({openaiApiKey: apiKey}, function() {
      console.log('API key saved');
    });
  }

  function resetFloatingWindow() {
    const apiKeyInput = document.getElementById('openai-api-key');
    const wordCountSelect = document.getElementById('summarizer-word-count');
    
    if (apiKeyInput) {
      apiKeyInput.value = '';
    }
    
    if (wordCountSelect) {
      wordCountSelect.value = '100'; // Set to default value
      localStorage.setItem('summarizerDefaultWordCount', '100');
    }
  }

  console.log("Content script loaded");

})();

async function retryWithBackoff(operation, maxRetries = 5, initialDelay = 2000) {
  let delay = initialDelay;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1 || !error.message.includes('Rate limit exceeded')) throw error;
      console.log(`Attempt ${i + 1} failed, retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      delay *= 2; // exponential backoff
    }
  }
}

// Add this CSS to your existing styles or inject it into the page
const style = document.createElement('style');
style.textContent = `
  .resize-handle {
    width: 10px;
    height: 10px;
    background-color: #ccc;
    position: absolute;
    right: 0;
    bottom: 0;
    cursor: se-resize;
  }
`;
document.head.appendChild(style);
