/**
 * Preset Test Image Samples Manager
 * Allows 1-click evaluation without uploading files
 */

export const SAMPLE_IMAGES = [
  {
    id: 'retriever',
    title: 'Retriever',
    category: 'Dog / Animal',
    url: '/samples/golden_retriever.jpg',
    expectedClass: 'Golden Retriever (MobileNet) / Dog (Custom CNN)'
  },
  {
    id: 'car',
    title: 'Sports Car',
    category: 'Automobile',
    url: '/samples/sports_car.jpg',
    expectedClass: 'Sports car (MobileNet) / Automobile (Custom CNN)'
  },
  {
    id: 'mug',
    title: 'Coffee Mug',
    category: 'Household',
    url: '/samples/coffee_mug.jpg',
    expectedClass: 'Coffee mug / Cup (MobileNet)'
  },
  {
    id: 'airliner',
    title: 'Airliner',
    category: 'Aviation',
    url: '/samples/airliner.jpg',
    expectedClass: 'Airliner (MobileNet) / Airplane (Custom CNN)'
  },
  {
    id: 'daisy',
    title: 'Daisy',
    category: 'Flora',
    url: '/samples/daisy.jpg',
    expectedClass: 'Daisy (MobileNet)'
  }
];

export function setupSampleGallery(containerEl, onSelectSample) {
  if (!containerEl) return;

  containerEl.innerHTML = SAMPLE_IMAGES.map(sample => `
    <div class="sample-card" data-sample-id="${sample.id}" title="${sample.title} - ${sample.category}">
      <img src="${sample.url}" alt="${sample.title}" class="sample-thumb" loading="lazy" />
      <div class="sample-caption">${sample.title}</div>
    </div>
  `).join('');

  containerEl.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      const sampleId = card.getAttribute('data-sample-id');
      const sample = SAMPLE_IMAGES.find(s => s.id === sampleId);
      if (sample && onSelectSample) {
        onSelectSample(sample);
      }
    });
  });
}
