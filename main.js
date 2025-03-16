// Function to fetch data from the API using a dynamic text query
function fetchDataFromApi(text) {
    // Build the API URL with the provided text
    const apiUrl = `http://127.0.0.1:8000/analyze?text=${encodeURIComponent(text)}`;
  
    // Fetch data from the API
    return fetch(apiUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();  // Parse the JSON response
      })
      .then(data => {
        return data;  // Return the parsed JSON data
      })
      .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;  // Rethrow the error so it can be handled by the caller
      });
  }
  

  async function logApiResponse() {
    try {
      const data = await fetchDataFromApi("kill yourself");
      console.log(data.safe_for_snowflake);
      console.log(data.offensive);
    } catch (error) {
      console.error('There was an error:', error);
    }
  }
  
  logApiResponse();
  