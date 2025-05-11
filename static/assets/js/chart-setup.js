let chartInstance = null;

function renderChart(allScores) {
  const ctx = document.getElementById('sentimentChart').getContext('2d');
  if (window.chartInstance) window.chartInstance.destroy();

  window.chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Negative', 'Neutral', 'Positive'],
      datasets: [{
        label: 'Sentiment Score',
        data: allScores,
        backgroundColor: ['#ef4444', '#facc15', '#22c55e']
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 1 }
      }
    }
  });
}

