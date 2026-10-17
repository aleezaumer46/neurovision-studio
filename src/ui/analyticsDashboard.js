/**
 * Module 11: Real-Time Telemetry & Analytics Dashboard
 * Interacts with the async FastAPI backend and renders Chart.js visualizations
 */
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

const API_BASE = 'http://localhost:8000';

export class AnalyticsDashboard {
  constructor() {
    this.barChartInstance = null;
    this.lineChartInstance = null;
    this.isBackendOnline = false;
  }

  /**
   * Log inference event to FastAPI backend (or local storage fallback)
   */
  async logInference(eventData) {
    const payload = {
      ...eventData,
      top_label: eventData.top_label ?? eventData.top_class ?? 'Unknown',
      top_class: eventData.top_class ?? eventData.top_label ?? 'Unknown',
      inference_time_ms: eventData.inference_time_ms ?? eventData.latency_ms ?? 0,
      latency_ms: eventData.latency_ms ?? eventData.inference_time_ms ?? 0,
      timestamp: eventData.timestamp ?? new Date().toISOString()
    };
    try {
      const res = await fetch(`${API_BASE}/api/log-inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) {
        this.isBackendOnline = true;
      }
    } catch (e) {
      // Offline fallback: save to localStorage
      this.isBackendOnline = false;
      this.saveLocalLog(payload);
    }
  }

  saveLocalLog(eventData) {
    try {
      const logs = JSON.parse(localStorage.getItem('neurovision_telemetry_logs') || '[]');
      logs.unshift({ ...eventData, timestamp: new Date().toISOString() });
      if (logs.length > 100) logs.pop();
      localStorage.setItem('neurovision_telemetry_logs', JSON.stringify(logs));
    } catch (e) {}
  }

  /**
   * Fetch aggregated analytics from FastAPI backend
   */
  async fetchAnalytics() {
    try {
      const res = await fetch(`${API_BASE}/api/analytics`, {
        signal: AbortSignal.timeout(3000)
      });
      if (res.ok) {
        this.isBackendOnline = true;
        return await res.json();
      }
    } catch (e) {
      this.isBackendOnline = false;
    }

    // Fallback: aggregate from localStorage
    return this.getLocalAnalytics();
  }

  getLocalAnalytics() {
    const logs = JSON.parse(localStorage.getItem('neurovision_telemetry_logs') || '[]');
    const total = logs.length;
    const avgLatency = total > 0 ? logs.reduce((a, b) => a + (b.latency_ms || 0), 0) / total : 0;

    const classCounts = {};
    const modelCounts = {};
    logs.forEach(l => {
      classCounts[l.top_class] = (classCounts[l.top_class] || 0) + 1;
      modelCounts[l.model_name] = (modelCounts[l.model_name] || 0) + 1;
    });

    const topClasses = Object.entries(classCounts)
      .map(([k, v]) => ({ class_name: k, count: v }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    const latencyTrends = logs.slice(0, 20).reverse().map((l, i) => ({
      point: i + 1,
      latency_ms: l.latency_ms || 20,
      model: l.model_name
    }));

    return {
      total_inferences: total,
      average_latency_ms: Math.round(avgLatency),
      min_latency_ms: total > 0 ? Math.min(...logs.map(l => l.latency_ms || 0)) : 0,
      max_latency_ms: total > 0 ? Math.max(...logs.map(l => l.latency_ms || 0)) : 0,
      top_classes: topClasses,
      latency_trends: latencyTrends,
      model_distribution: modelCounts,
      backend_distribution: { webgl: total },
      recent_events: logs.slice(0, 10)
    };
  }

  /**
   * Render Top Classes Bar Chart
   */
  renderDistributionChart(canvasEl, analyticsData) {
    if (!canvasEl) return;

    if (this.barChartInstance) {
      this.barChartInstance.destroy();
    }

    const labels = analyticsData.top_classes?.length > 0
      ? analyticsData.top_classes.map(c => c.class_name.substring(0, 14))
      : ['No Inferences Yet'];
    const data = analyticsData.top_classes?.length > 0
      ? analyticsData.top_classes.map(c => c.count)
      : [0];

    const ctx = canvasEl.getContext('2d');
    this.barChartInstance = new Chart(ctx, {
      type: 'bar',
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
          y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
        }
      },
      data: {
        labels,
        datasets: [{
          label: 'Inference Frequency',
          data,
          backgroundColor: 'rgba(6, 182, 212, 0.65)',
          borderColor: '#06b6d4',
          borderWidth: 2,
          borderRadius: 6
        }]
      },
    });
  }

  /**
   * Render Real-Time Latency Trends Line Chart
   */
  renderLatencyTrendChart(canvasEl, analyticsData) {
    if (!canvasEl) return;

    if (this.lineChartInstance) {
      this.lineChartInstance.destroy();
    }

    const trends = analyticsData.latency_trends || [];
    const labels = trends.map(t => `#${t.point}`);
    const data = trends.map(t => t.latency_ms);

    const ctx = canvasEl.getContext('2d');
    this.lineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Latency (ms)',
          data,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.15)',
          borderWidth: 3,
          tension: 0.35,
          fill: true,
          pointBackgroundColor: '#06b6d4',
          pointBorderColor: '#fff',
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: {
            beginAtZero: true,
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        }
      }
    });
  }
}

export const analyticsDashboard = new AnalyticsDashboard();
