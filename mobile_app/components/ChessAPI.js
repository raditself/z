
const API_URL = 'http://localhost:8000';  // Replace with your actual API URL

export const makeMove = async (fen, move) => {
  try {
    const response = await fetch(`${API_URL}/make_move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ fen, move }),
    });

    if (!response.ok) {
      throw new Error('API request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error making API call:', error);
    return null;
  }
};
