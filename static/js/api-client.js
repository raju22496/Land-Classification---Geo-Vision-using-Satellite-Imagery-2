/**
 * API Client for Land Classification System
 */

class LandClassificationApi {
  /**
   * Create a new API client instance
   * @param {string} baseUrl - API base URL (defaults to current host)
   */
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl || window.location.origin;
  }

  /**
   * Make a GET request to the API
   * @param {string} endpoint - API endpoint
   * @param {object} params - Query parameters
   * @returns {Promise} API response
   */
  async get(endpoint, params = {}) {
    const url = new URL(`${this.baseUrl}${endpoint}`);

    // Add query parameters
    Object.keys(params).forEach((key) => {
      url.searchParams.append(key, params[key]);
    });

    try {
      const response = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });

      return this._handleResponse(response);
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Make a POST request to the API
   * @param {string} endpoint - API endpoint
   * @param {object|FormData} data - Request data
   * @param {boolean} isFormData - Whether data is FormData
   * @returns {Promise} API response
   */
  async post(endpoint, data = {}, isFormData = false) {
    const url = `${this.baseUrl}${endpoint}`;
    const options = {
      method: "POST",
      headers: {},
    };

    if (isFormData) {
      // If data is FormData, do not set Content-Type
      // The browser will set it automatically with the correct boundary
      options.body = data;
    } else {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(url, options);
      return this._handleResponse(response);
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Upload a file for prediction
   * @param {File} file - File object to upload
   * @returns {Promise} API response
   */
  async predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    return this.post("/api/predict", formData, true);
  }

  /**
   * Upload multiple files for batch prediction
   * @param {Array<File>} files - Array of File objects
   * @returns {Promise} API response
   */
  async batchPredict(files) {
    const formData = new FormData();

    files.forEach((file, index) => {
      formData.append("files[]", file);
    });

    return this.post("/api/batch-predict", formData, true);
  }

  /**
   * Handle API response
   * @param {Response} response - Fetch Response object
   * @returns {Promise} Parsed response data
   * @private
   */
  async _handleResponse(response) {
    const contentType = response.headers.get("Content-Type") || "";

    if (contentType.includes("application/json")) {
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP error ${response.status}`);
      }

      return data;
    } else {
      const text = await response.text();

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      return text;
    }
  }
}

// Create a global instance
const api = new LandClassificationApi();
