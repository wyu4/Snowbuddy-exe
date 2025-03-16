// content.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeAndHighlight") {
      console.log("Received message in content script to analyze and highlight.");
  
      // Example of text analysis logic
      const bodyText = document.body.innerText;
      const sentences = bodyText.split('.'); // Simple sentence splitting
  
      const offensiveWords = ["offensiveWord1", "offensiveWord2"]; // Example offensive words
      let highlightedText = bodyText;
  
      sentences.forEach(sentence => {
        offensiveWords.forEach(word => {
          if (sentence.includes(word)) {
            // Highlight offensive sentences (simplified example)
            highlightedText = highlightedText.replace(sentence, `<span style="background-color: red">${sentence}</span>`);
          }
        });
      });
  
      // Replace body content with highlighted text (for demonstration)
      document.body.innerHTML = highlightedText;
  
      sendResponse({ success: true, highlightedText });
    }
  
    // Return true to indicate that the response is sent asynchronously
    return true;
  });
  