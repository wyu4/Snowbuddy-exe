// background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeAndHighlight") {
      console.log("Received message in background: analyzing content...");
      
      // You can perform background tasks here if needed, like fetching data from a server or manipulating other things.
      
      sendResponse({ success: true });
    }
    
    // Return true to indicate the response will be sent asynchronously
    return true;
  });
  