function getEmailBody() {
    // Try compose window first
    const composeBody = document.querySelector('div[role="textbox"]');
    if (composeBody) return composeBody.innerText;
    
    // Try reading view
    const viewBody = document.querySelector('div[role="article"]');
    return viewBody ? viewBody.innerText : '';
  }
  
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getEmailBody') {
      sendResponse({ emailBody: getEmailBody() });
    }
    return true;
  });