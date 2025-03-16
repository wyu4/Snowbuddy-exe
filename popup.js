document.addEventListener('DOMContentLoaded', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['contentScript.js']
      }, () => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'getEmailBody' }, (response) => {
          // Set username from Gmail account button
          document.getElementById('greeting').textContent = `Dear ${response?.username || 'User'},`;
  
          if (chrome.runtime.lastError) {
            document.getElementById('responseText').textContent = 'Error: Please refresh Gmail and try again.';
            return;
          }
          
          const emailText = response?.emailBody || '';
          if (!emailText.trim()) {
            document.getElementById('responseText').textContent = 'No email content found.';
            return;
          }
  
          fetch(`http://127.0.0.1:8000/analyze?text=${encodeURIComponent(emailText)}`)
            .then((res) => res.json())
            .then((data) => displayResults(data))
            .catch((error) => {
              document.getElementById('responseText').textContent = 'API Error: ' + error.message;
            });
        });
      });
    });
  });
  
  function displayResults(data) {
    const responseElement = document.getElementById('responseText');
    const progressBar = document.getElementById('progressBar');
    const statusText = document.getElementById('statusText');
    const percentageText = document.getElementById('percentageText');
  
    responseElement.textContent = data.response;
    
    // Correct score handling
    const offensiveScore = data.safe_for_snowflake;
    const percentage = (offensiveScore * 100).toFixed(1);
  
    statusText.textContent = 'Offensive';
    statusText.style.color = '#f44336';
    
    percentageText.textContent = `${percentage}%`;
    percentageText.style.color = '#f44336';
  
    progressBar.style.width = `${percentage}%`;
    progressBar.style.backgroundColor = '#f44336';
  }