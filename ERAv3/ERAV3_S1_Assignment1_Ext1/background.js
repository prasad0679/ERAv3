chrome.action.onClicked.addListener((tab) => {
  // Check if we can inject scripts into this tab
  if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://")) {
    console.log("Cannot inject scripts into chrome:// or edge:// URLs");
    return;
  }

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  }).then(() => {
    chrome.tabs.sendMessage(tab.id, { action: "toggleFloatingWindow" });
  }).catch(error => {
    console.error("Error executing script:", error);
  });

  chrome.scripting.insertCSS({
    target: { tabId: tab.id },
    files: ['styles.css']
  }).catch(error => {
    console.error("Error injecting CSS:", error);
  });
});
