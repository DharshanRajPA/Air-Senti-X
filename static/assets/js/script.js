document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('analyzeBtn').addEventListener('click', analyzeTweet);
});

async function analyzeTweet() {
  const tweet = document.getElementById('tweetInput').value.trim();
  if (!tweet) return;

  document.getElementById('loader').classList.remove('hidden');
  document.getElementById('resultSection').classList.add('hidden');

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: tweet })
    });

    const data = await response.json();
    document.getElementById('loader').classList.add('hidden');

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    document.getElementById('sentiment').textContent = data.label || 'N/A';
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
    document.getElementById('tone').textContent = data.emotional_tone || 'neutral';
    document.getElementById('urgency').textContent = (data.urgency_score || 0.0).toFixed(2);

    renderChart(data.all_scores);
    document.getElementById('resultSection').classList.remove('hidden');

  } catch (err) {
    document.getElementById('loader').classList.add('hidden');
    console.error("Request failed:", err);
    alert("Request failed. Check if the backend is running.");
  }
}
