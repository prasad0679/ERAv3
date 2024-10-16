(function() {
  let summarizerActive = false;
  let currentUtterance = null;

  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggleFloatingWindow") {
      toggleFloatingWindow();
    }
  });

  function initializeSummarizer() {
    const floatingWindow = document.getElementById('summarizer-floating-window');
    const summarizerButton = document.getElementById('summarizer-button');
    const wordCountSelect = document.getElementById('summarizer-word-count');
    const summarizerClose = document.getElementById('summarizer-close');

    makeDraggable(floatingWindow);

    summarizerButton.addEventListener('click', handleSummarizerClick);
    summarizerClose.addEventListener('click', closeFloatingWindow);
    
    // Ensure the dropdown is clickable and not affected by the draggable behavior
    wordCountSelect.addEventListener('mousedown', (e) => {
      e.stopPropagation();
    });

    wordCountSelect.addEventListener('change', updateDefaultWordCount);

    // Set the initial value from localStorage
    const savedWordCount = localStorage.getItem('summarizerDefaultWordCount') || '100';
    wordCountSelect.value = savedWordCount;

    // Debug log
    console.log('Initializing summarizer with word count:', savedWordCount);
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
      closeSummaryWindow();
      closeFloatingWindow();
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

  function summarizeText(text, length) {
    const apiKey = '<add API key here>'; // This should be replaced with a valid API key
    const apiUrl = 'https://api.openai.com/v1/chat/completions';

    const prompt = `Summarize the following text in approximately ${length} words:\n\n${text}`;

    return fetch(apiUrl, {
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
    })
    .then(response => {
      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Invalid API key. Please check your OpenAI API key and try again.');
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }
      return response.json();
    })
    .then(data => data.choices[0].message.content.trim())
    .catch(error => {
      console.error('Error calling OpenAI API:', error);
      return `An error occurred while generating the summary: ${error.message}. Please check your API key and try again.`;
    });
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

  function toggleFloatingWindow() {
    if (document.getElementById('summarizer-floating-window')) {
      closeFloatingWindow();
    } else {
      const floatingWindow = document.createElement('div');
      floatingWindow.id = 'summarizer-floating-window';
      floatingWindow.innerHTML = `
        <div id="summarizer-handle" class="draggable-handle">
          <button id="summarizer-button">S</button>
          <div class="word-count-container">
            <select id="summarizer-word-count">
              ${generateWordCountOptions()}
            </select>
            <span>words</span>
          </div>
          <button id="summarizer-close">C</button>
        </div>
      `;
      document.body.appendChild(floatingWindow);
      initializeSummarizer();
    }
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
      floatingWindow.remove();
    }
    closeSummaryWindow();
  }

  // Expose the toggleFloatingWindow function to the global scope
  window.toggleFloatingWindow = toggleFloatingWindow;

})();
