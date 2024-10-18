(function() {
  let summarizerActive = false;
  let currentUtterance = null;

  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "openFloatingWindow") {
      openFloatingWindow();
    }
  });

  function initializeSummarizer() {
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

  function handleSummarizerClick() {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText.length > 0) {
      summarizerActive = true;
      summarizeSelectedText();
    } else {
      showMessage("Please select the text for summarization and then click the button");
    }
  }

  function summarizeSelectedText() {
    const selectedText = window.getSelection().toString().trim();
    const summaryLength = parseInt(document.getElementById('summarizer-word-count').value);
    showLoadingIndicator();
    summarizeText(selectedText, summaryLength).then(summary => {
      hideLoadingIndicator();
      showSummaryWindow(summary);
    }).catch(error => {
      hideLoadingIndicator();
      console.error('Error generating summary:', error);
      showSummaryWindow('An error occurred while generating the summary. Please try again.');
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
    `;
    document.body.appendChild(summaryWindow);

    makeDraggable(summaryWindow);
    makeResizable(summaryWindow);

    document.getElementById('summarizer-summary-close').addEventListener('click', closeSummaryWindow);
    document.getElementById('summarizer-read-aloud').addEventListener('click', () => readAloud(summary));
    document.getElementById('summarizer-export').addEventListener('click', () => exportSummary(summary));
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
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    const handle = element.querySelector('.draggable-handle') || element;
    
    handle.onmousedown = dragMouseDown;

    function dragMouseDown(e) {
      e.preventDefault();
      pos3 = e.clientX;
      pos4 = e.clientY;
      document.onmouseup = closeDragElement;
      document.onmousemove = elementDrag;
    }

    function elementDrag(e) {
      e.preventDefault();
      pos1 = pos3 - e.clientX;
      pos2 = pos4 - e.clientY;
      pos3 = e.clientX;
      pos4 = e.clientY;
      element.style.top = (element.offsetTop - pos2) + "px";
      element.style.left = (element.offsetLeft - pos1) + "px";
    }

    function closeDragElement() {
      document.onmouseup = null;
      document.onmousemove = null;
    }
  }

  function makeResizable(element) {
    const resizer = document.createElement('div');
    resizer.className = 'resizer';
    resizer.style.width = '10px';
    resizer.style.height = '10px';
    resizer.style.background = 'red';
    resizer.style.position = 'absolute';
    resizer.style.right = '0';
    resizer.style.bottom = '0';
    resizer.style.cursor = 'se-resize';
    element.appendChild(resizer);

    resizer.addEventListener('mousedown', initResize, false);

    function initResize(e) {
      window.addEventListener('mousemove', resize, false);
      window.addEventListener('mouseup', stopResize, false);
    }

    function resize(e) {
      element.style.width = (e.clientX - element.offsetLeft) + 'px';
      element.style.height = (e.clientY - element.offsetTop) + 'px';
    }

    function stopResize(e) {
      window.removeEventListener('mousemove', resize, false);
      window.removeEventListener('mouseup', stopResize, false);
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

  async function summarizeText(text, length) {
    const apiKey = document.getElementById('openai-api-key').value;
    if (!apiKey) {
      return 'Please enter your OpenAI API key in the text field above the "S" button.';
    }

    try {
      const apiUrl = 'https://api.openai.com/v1/chat/completions';
      const prompt = `Summarize the following text in approximately ${length} words:\n\n${text}`;

      const apiResponse = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [
            { role: 'system', content: 'You are a helpful assistant that summarizes text.' },
            { role: 'user', content: prompt }
          ],
          max_tokens: length * 2,
          n: 1,
          stop: null,
          temperature: 0.7,
        })
      });

      if (!apiResponse.ok) {
        if (apiResponse.status === 401) {
          throw new Error('Invalid API key. Please check your OpenAI API key and try again.');
        } else {
          throw new Error(`HTTP error! status: ${apiResponse.status}`);
        }
      }

      const data = await apiResponse.json();
      return data.choices[0].message.content.trim();
    } catch (error) {
      console.error('Error calling OpenAI API:', error);
      return `An error occurred while generating the summary: ${error.message}. Please check your API key and try again.`;
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
