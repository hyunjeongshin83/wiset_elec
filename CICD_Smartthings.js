const express = require('express');
const app = express();
const smartthings = require('smartthings');
const axios = require('axios'); // Import Axios library for making HTTP requests

// SmartThings Configuration
const smartThingsConfig = {
  accessToken: 'YOUR_SMARTTHINGS_ACCESS_TOKEN',
  apiUrl: 'https://api.smartthings.com/v1',
  lampDeviceId: 'YOUR_LAMP_DEVICE_ID'
};

// SmartThings API Client
const smartThingsClient = new smartthings.ApiClient(smartThingsConfig.accessToken, smartThingsConfig.apiUrl);

// Server Configuration
const serverConfig = {
  baseUrl: 'https://your-server.com/api', // Replace with your server base URL
  fireRiskEndpoint: '/fire-risk' // Replace with your server's fire risk endpoint
};

// Middleware to parse request body
app.use(express.json());

// Route to turn the lamp on
app.put('/on', (req, res) => {
  smartThingsClient.devices.commands(smartThingsConfig.lampDeviceId, 'switch', 'on')
    .then(() => {
      console.log('Lamp turned on');
      res.sendStatus(200);
    })
    .catch((error) => {
      console.error('Error turning lamp on:', error);
      res.status(500).send('Error turning lamp on');
    });
});

// Route to turn the lamp off
app.put('/off', (req, res) => {
  smartThingsClient.devices.commands(smartThingsConfig.lampDeviceId, 'switch', 'off')
    .then(() => {
      console.log('Lamp turned off');
      res.sendStatus(200);
    })
    .catch((error) => {
      console.error('Error turning lamp off:', error);
      res.status(500).send('Error turning lamp off');
    });
});

// Route to receive fire risk status from the server
app.post('/fire-risk', (req, res) => {
  const fireRiskStatus = req.body.status;

  // Handle the fire risk status
  handleFireRiskStatus(fireRiskStatus);

  res.sendStatus(200);
});

// Function to handle the fire risk status
function handleFireRiskStatus(status) {
  console.log(`Fire risk status received: ${status}`);

  // Add your logic here to handle the fire risk status
  // For example, you could turn the lamp on or off based on the status
  if (status === 'high') {
    smartThingsClient.devices.commands(smartThingsConfig.lampDeviceId, 'switch', 'on')
      .then(() => {
        console.log('Lamp turned on due to high fire risk');
      })
      .catch((error) => {
        console.error('Error turning lamp on:', error);
      });
  } else if (status === 'low') {
    smartThingsClient.devices.commands(smartThingsConfig.lampDeviceId, 'switch', 'off')
      .then(() => {
        console.log('Lamp turned off due to low fire risk');
      })
      .catch((error) => {
        console.error('Error turning lamp off:', error);
      });
  }
}

// Function to fetch fire risk status from the server
async function fetchFireRiskStatus() {
  try {
    const response = await axios.get(`${serverConfig.baseUrl}${serverConfig.fireRiskEndpoint}`);
    const fireRiskStatus = response.data.status;
    handleFireRiskStatus(fireRiskStatus);
  } catch (error) {
    console.error('Error fetching fire risk status:', error);
  }
}

// Fetch fire risk status from the server periodically (e.g., every 5 minutes)
setInterval(fetchFireRiskStatus, 5 * 60 * 1000);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});