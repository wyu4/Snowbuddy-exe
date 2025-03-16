function getUsername() {
  const accountButton = document.querySelector('a.gb_B.gb_Za.gb_0');
  if (!accountButton) return 'User';
  
  const label = accountButton.getAttribute('aria-label') || '';
  const match = label.match(/Account:\s([^(]+)/);
  
  if (match && match[1]) {
    return match[1].trim().split(' ')[0];
  }
  return 'User';
}

function getEmailBody() {
  const composeBody = document.querySelector('div[role="textbox"]');
  const viewBody = document.querySelector('div[role="article"]');
  return {
    emailBody: (composeBody || viewBody)?.innerText || '',
    username: getUsername()
  };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getEmailBody') {
    sendResponse(getEmailBody());
  }
  return true;
});