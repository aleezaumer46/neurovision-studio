/**
 * Information & Technical Architecture Modal
 */

export function setupInfoModal() {
  const modalBackdrop = document.getElementById('infoModal');
  const openBtn = document.getElementById('btnOpenInfo');
  const closeBtn = document.getElementById('btnCloseInfo');
  const pythonGuideBtn = document.getElementById('btnOpenPythonGuide');

  function openModal(activeTab = 'architecture') {
    if (!modalBackdrop) return;
    modalBackdrop.classList.add('open');
    switchTab(activeTab);
  }

  function closeModal() {
    if (!modalBackdrop) return;
    modalBackdrop.classList.remove('open');
  }

  function switchTab(tabId) {
    document.querySelectorAll('.modal-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.modal-tab-pane').forEach(pane => {
      pane.style.display = pane.id === `tab-${tabId}` ? 'block' : 'none';
    });
  }

  if (openBtn) openBtn.addEventListener('click', () => openModal('architecture'));
  if (pythonGuideBtn) pythonGuideBtn.addEventListener('click', () => openModal('python'));
  if (closeBtn) closeBtn.addEventListener('click', closeModal);

  modalBackdrop?.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) closeModal();
  });

  document.querySelectorAll('.modal-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}
