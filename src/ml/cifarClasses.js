/**
 * Vision Taxonomy & Class Definitions
 * 10-Class Transportation & Travel Vehicle Classification
 */

export const VEHICLE_CLASSES = [
  { id: 0, name: 'Airplane', icon: '✈️', description: 'Commercial aircraft, passenger jetliners, gliders, and light airplanes' },
  { id: 1, name: 'Automobile / Car', icon: '🚗', description: 'Sedans, sports cars, coupés, convertibles, and consumer passenger cars' },
  { id: 2, name: 'Bus', icon: '🚌', description: 'Public transit buses, school buses, motorcoaches, and double-deckers' },
  { id: 3, name: 'Train', icon: '🚆', description: 'High-speed passenger trains, locomotives, light rail, and subways' },
  { id: 4, name: 'Ship / Boat', icon: '🚢', description: 'Ocean liners, cargo ships, ferries, yachts, speedboats, and watercraft' },
  { id: 5, name: 'Truck', icon: '🚚', description: 'Heavy transport lorries, semi-trailers, haulers, dump trucks, and tow trucks' },
  { id: 6, name: 'Motorcycle', icon: '🏍️', description: 'Motorbikes, sport motorcycles, cruisers, mopeds, and scooters' },
  { id: 7, name: 'Bicycle', icon: '🚲', description: 'Road bicycles, mountain bikes, urban commuter cycles, and tandem bikes' },
  { id: 8, name: 'Helicopter', icon: '🚁', description: 'Rotorcraft, transport choppers, civil helicopters, and medical air rescue' },
  { id: 9, name: 'Van', icon: '🚐', description: 'Minivans, passenger transit vans, cargo delivery vans, and campervans' }
];

export const MODEL_SPECS = {
  mobilenet: {
    name: 'MobileNetV2 (ImageNet)',
    type: 'Convolutional Neural Network (Inverted Residuals)',
    classesCount: 1000,
    inputResolution: '224 × 224 × 3',
    depth: '53 Layers',
    parameters: '3.5 Million',
    quantization: 'Float32 / Float16 WebGL',
    description: 'Pre-trained deep CNN using depthwise separable convolutions for high accuracy object classification.'
  },
  custom_cnn: {
    name: 'Transportation & Vehicle Vision CNN',
    type: 'Convolutional Neural Network (Conv2D + BatchNorm + GAP)',
    classesCount: 10,
    inputResolution: '224 × 224 × 3',
    depth: '12 Layers',
    parameters: '103,050',
    quantization: 'Float32 Browser Weights',
    description: 'Custom client-side CNN trained for 10 Travel & Transportation vehicle classes with Out-of-Distribution uncertainty estimation.'
  }
};
