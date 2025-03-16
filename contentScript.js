// contentScript.js
let snowflakeWindow = null;
let updateInterval = null;
let isUpdating = false;

let isDragging = false;
let offset = [0, 0];
let isMaximized = false;
let isMinimized = false;

var timerUpdate = 500;

function createSnowflakeWindow() {
  if (document.getElementById('snowflake-window')) return;

  // Create window structure using your popup.html content
  const windowHTML = `
    <div id="snowflake-window">
      <div id="titlebar">
        <span>Snowflake Preventive</span>
        <div class="window-controls">
          <button class="window-control minimize">−</button>
          <button class="window-control maximize">□</button>
          <button class="window-control close">×</button>
        </div>
      </div>
      <div class="window-content">
        <h1>Snowflake Preventive</h1>
        <div class="email-container">
          <div class="email-header">
            <div id="greeting">Dear User,</div>
          </div>
          <div class="email-content">
            <div id="responseText">Analyzing your email content...</div>
          </div>
          <div class="email-signature">
            <div>Sincerely,</div>
            <div>Snowflake Detector</div>
          </div>
        </div>
        <div class="status-container">
          <div id="statusText" class="status-text"></div>
          <div id="percentageText" class="status-text"></div>
        </div>
        <div class="progress-container">
          <div id="progressBar" class="progress-bar"></div>
        </div>
      </div>
    </div>
  `;

  const wrapper = document.createElement('div');
  wrapper.innerHTML = windowHTML;
  snowflakeWindow = wrapper.firstElementChild;
  document.body.appendChild(snowflakeWindow);

  // Add window controls functionality
  addWindowControls();
  // Add dragging functionality
  setupDragging();
  // Load initial content
  loadContent();

  // start auto update
  startAutoUpdate();
}

function startAutoUpdate() {
  stopAutoUpdate(); // Clear any existing interval
  updateInterval = setInterval(() => {
    if (!isUpdating) loadContent();
  }, timerUpdate);
}

function stopAutoUpdate() {
  if (updateInterval) {
    clearInterval(updateInterval);
    updateInterval = null;
  }
}


function addWindowControls() {
  const titlebar = snowflakeWindow.querySelector('#titlebar');
  const minimizeBtn = snowflakeWindow.querySelector('.minimize');
  const maximizeBtn = snowflakeWindow.querySelector('.maximize');
  const closeBtn = snowflakeWindow.querySelector('.close');

  minimizeBtn.addEventListener('click', () => {
    isMinimized = !isMinimized;
    const content = snowflakeWindow.querySelector('.window-content');
    content.style.display = isMinimized ? 'none' : 'block';
  });

  maximizeBtn.addEventListener('click', () => {
    isMaximized = !isMaximized;
    if (isMaximized) {
      snowflakeWindow.style.width = '95%';
      snowflakeWindow.style.height = '95%';
      snowflakeWindow.style.left = '2.5%';
      snowflakeWindow.style.top = '2.5%';
    } else {
      snowflakeWindow.style.width = '300px';
      snowflakeWindow.style.height = 'auto';
      snowflakeWindow.style.left = '20px';
      snowflakeWindow.style.top = '20px';
    }
  });

  closeBtn.addEventListener('click', () => {
    snowflakeWindow.remove();
    snowflakeWindow = null;
    stopAutoUpdate();
  });
}

function setupDragging() {
  const titlebar = snowflakeWindow.querySelector('#titlebar');

  titlebar.addEventListener('mousedown', (e) => {
    if (e.target.closest('.window-control')) return;
    isDragging = true;
    offset = [
      snowflakeWindow.offsetLeft - e.clientX,
      snowflakeWindow.offsetTop - e.clientY
    ];
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    snowflakeWindow.style.left = `${e.clientX + offset[0]}px`;
    snowflakeWindow.style.top = `${e.clientY + offset[1]}px`;
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
  });
}

function loadContent() {

  if (isUpdating) return;
  isUpdating = true;

  const emailData = getEmailBody();
  document.getElementById('greeting').textContent = `Dear ${emailData.username || 'User'},`;

  if (!emailData.emailBody.trim()) {
    document.getElementById('responseText').textContent = 'No email content found.';
    return;
  }
  
  fetch(`http://127.0.0.1:8000/analyze?text=${encodeURIComponent(emailData.emailBody)}`)
    .then((res) => res.json())
    .then((data) =>{
      displayResults(data);
      isUpdating = false;
    } )
    .catch((error) => {
      document.getElementById('responseText').textContent = 'API Error: ' + error.message;
      isUpdating = false;
    });
}



function displayResults(data) {
  console.log(data);
  const responseElement = document.getElementById('responseText');
  const progressBar = document.getElementById('progressBar');
  const statusText = document.getElementById('statusText');
  const percentageText = document.getElementById('percentageText');

  responseElement.textContent = data.response;
  const offensiveScore = data.safe_for_snowflake;
  const percentage = (offensiveScore * 100).toFixed(1);

  statusText.textContent = 'Offensive';
  statusText.style.color = '#f44336';
  percentageText.textContent = `${percentage}%`;
  percentageText.style.color = '#f44336';
  progressBar.style.width = `${percentage}%`;
  progressBar.style.backgroundColor = '#f44336';
}

// Original content script functions
function getUsername() {
  const accountButton = document.querySelector('a.gb_B.gb_Za.gb_0');
  if (!accountButton) return 'User';
  const label = accountButton.getAttribute('aria-label') || '';
  const match = label.match(/Account:\s([^(]+)/);
  return match && match[1] ? match[1].trim().split(' ')[0] : 'User';
}

function getEmailBody() {
  const composeBody = document.querySelector('div[role="textbox"]');
  const viewBody = document.querySelector('div[role="article"]');
  return {
    emailBody: (composeBody || viewBody)?.innerText || '',
    username: getUsername()
  };
}

// Toggle window visibility
chrome.runtime.onMessage.addListener((request) => {
  if (request.action === 'toggleWindow') {
    if (!snowflakeWindow) {
      createSnowflakeWindow();
    } else {
      const wasVisible = snowflakeWindow.style.display !== 'none';
      snowflakeWindow.style.display = wasVisible ? 'none' : 'block';
      
      if (wasVisible) {
        stopAutoUpdate();
      } else {
        startAutoUpdate();
      }
    }
  }
  return true;
});