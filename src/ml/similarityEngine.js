/**
 * Module 5: Visual Similarity Search (Reverse Image Lookup)
 * Feature Vector Embeddings via GAP & Cosine Similarity Ranking
 */
import * as tf from '@tensorflow/tfjs';

export class VisualSimilarityEngine {
  constructor() {
    this.galleryIndex = [];
    this.isIndexed = false;
  }

  /**
   * Extracts a normalized 1D L2-embedding vector for any image
   */
  extractEmbedding(mobilenetModel, mediaElement) {
    return tf.tidy(() => {
      // Extract intermediate GAP representation
      const intermediate = mobilenetModel.infer(mediaElement, true);
      // Flatten to 1D
      const flat = intermediate.squeeze();
      // Compute L2 norm
      const norm = flat.norm();
      // Normalize
      const normalized = flat.div(norm);
      return Array.from(normalized.dataSync());
    });
  }

  /**
   * Cosine similarity between two unit vectors: dot product
   */
  cosineSimilarity(vecA, vecB) {
    let dot = 0;
    const len = Math.min(vecA.length, vecB.length);
    for (let i = 0; i < len; i++) {
      dot += vecA[i] * vecB[i];
    }
    return Math.max(0, Math.min(1, dot));
  }

  /**
   * Initializes reference gallery with pre-indexed sample embeddings
   */
  async buildIndex(mobilenetModel, sampleImages) {
    this.galleryIndex = [];

    for (const sample of sampleImages) {
      try {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = sample.url;
        });

        const vector = this.extractEmbedding(mobilenetModel, img);
        this.galleryIndex.push({
          id: sample.id,
          title: sample.title,
          category: sample.category,
          url: sample.url,
          vector
        });
      } catch (err) {
        console.warn(`Could not index sample ${sample.title}:`, err);
      }
    }

    this.isIndexed = true;
    return this.galleryIndex.length;
  }

  /**
   * Searches the indexed gallery for the most visually similar items
   */
  search(mobilenetModel, queryMedia, topK = 5) {
    if (this.galleryIndex.length === 0) {
      return [];
    }

    const queryVector = this.extractEmbedding(mobilenetModel, queryMedia);

    const scored = this.galleryIndex.map(item => {
      const similarity = this.cosineSimilarity(queryVector, item.vector);
      return {
        ...item,
        similarity,
        similarityPct: Math.round(similarity * 100)
      };
    });

    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
  }
}

export const similarityEngine = new VisualSimilarityEngine();
