// popup.js
document.getElementById("analyzeBtn").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tabId = tabs[0].id;
  
      console.log("Sending message to content script.");
      
      chrome.tabs.sendMessage(tabId, { action: "analyzeAndHighlight" }, (response) => {
        if (chrome.runtime.lastError) {
          console.error("Error sending message to content script:", chrome.runtime.lastError.message);
        } else {
          console.log("Message sent successfully.");
          if (response && response.highlightedText) {
            console.log("Highlighted text:", response.highlightedText);
          }
        }
      });
    });
  });
  