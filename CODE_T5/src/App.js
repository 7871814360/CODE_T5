import React, { useState } from 'react';

function App() {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('java');
  const [convertedCode, setConvertedCode] = useState('');
  const [error, setError] = useState(null); // To track errors

  const handleCodeChange = (e) => {
    setCode(e.target.value);
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleConvert = async () => {
    setError(null); // Reset error before making a new request
    try {
      const response = await fetch('http://localhost:5000/convert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code, language }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setConvertedCode(data.convertedCode);
    } catch (err) {
      setError('Failed to fetch converted code. Please try again.');
      console.error('Error:', err); // Log the error for debugging
    }
  };

  return (
    <div>
      <h1>Code Converter</h1>
      <textarea
        rows="10"
        cols="50"
        placeholder="Enter Python code here..."
        value={code}
        onChange={handleCodeChange}
      />
      <br />
      <label>Select Language:</label>
      <select value={language} onChange={handleLanguageChange}>
        <option value="java">Java</option>
        <option value="c">C</option>
        <option value="cpp">C++</option>
        <option value="javascript">JavaScript</option>
      </select>
      <br />
      <button onClick={handleConvert}>Convert</button>

      {error && <p style={{ color: 'red' }}>{error}</p>} {/* Display error if it occurs */}
      
      <h2>Converted Code</h2>
      <pre>{convertedCode}</pre>
    </div>
  );
}

export default App;
