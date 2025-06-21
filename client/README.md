# 🧠 Lenda Brainwave Blockchain Frontend

A React-based frontend application for visualizing brainwave-based blockchain data from EEG devices and Raspberry Pi systems.

## Features

### 🎯 Real-time Brainwave Analysis
- **Multi-frequency Band Visualization**: Displays Alpha, Beta, Theta, Delta, and Gamma waves
- **Live Data Streaming**: Real-time updates from EEG devices
- **Interactive Charts**: Built with Recharts for smooth, responsive visualizations
- **Frequency Band Cards**: Current values and ranges for each brainwave type

### 🔗 Obsidian-Style Node Network
- **Blockchain Node Visualization**: Force-directed graph showing node connections
- **Interactive Node Selection**: Click nodes to view detailed information
- **Dynamic Node Sizing**: Node size reflects brainwave values
- **Color-coded Types**: Different colors for different brainwave frequency types
- **Network Statistics**: Real-time stats on nodes, connections, and blockchain state

### 🎨 Modern UI/UX
- **Glassmorphism Design**: Modern glass-like interface with blur effects
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Dark Theme**: Optimized for low-light environments
- **Smooth Animations**: CSS transitions and micro-interactions

## Technology Stack

- **React 18** with TypeScript
- **Styled Components** for styling
- **Recharts** for data visualization
- **React Force Graph 2D** for node network visualization
- **D3.js** for advanced graphics

## Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
├── components/
│   ├── BrainwaveChart.tsx    # Real-time brainwave visualization
│   └── ObsidianNodes.tsx     # Blockchain node network
├── App.tsx                   # Main application component
├── App.css                   # Global styles
└── index.tsx                 # Application entry point
```

## Data Format

### Brainwave Data
```typescript
interface BrainwaveData {
  timestamp: Date;
  alpha: number;    // 8-13 Hz
  beta: number;     // 13-30 Hz
  theta: number;    // 4-8 Hz
  delta: number;    // 0.5-4 Hz
  gamma: number;    // 30-100 Hz
}
```

### Blockchain Nodes
```typescript
interface Node {
  id: string;
  label: string;
  type: string;           // brainwave frequency type
  value: number;          // frequency value in Hz
  timestamp: Date;
  connections: string[];  // connected node IDs
}
```

## Integration with Backend

The frontend is designed to work with a blockchain backend that:
- Receives HTTP requests from EEG devices
- Processes brainwave data from Raspberry Pi systems
- Builds a blockchain based on biometric data
- Provides real-time data updates via WebSocket or HTTP polling

### API Endpoints (Expected)
- `GET /api/brainwave/current` - Current brainwave readings
- `GET /api/blockchain/nodes` - Blockchain node data
- `GET /api/blockchain/status` - Blockchain status and statistics

## Customization

### Styling
- Modify `App.css` for global styles
- Update styled components in individual files for component-specific styling
- Theme colors can be adjusted in the styled components

### Data Sources
- Update the data generation logic in `App.tsx` to connect to your backend
- Modify the polling interval for real-time updates
- Add WebSocket support for live data streaming

### Chart Configuration
- Adjust chart properties in `BrainwaveChart.tsx`
- Modify node visualization settings in `ObsidianNodes.tsx`
- Customize colors and styling for different brainwave types

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Code Style
- TypeScript for type safety
- Functional components with hooks
- Styled components for CSS-in-JS
- ESLint and Prettier for code formatting

## Deployment

1. Build the application:
```bash
npm run build
```

2. Deploy the `build` folder to your hosting service:
   - Netlify
   - Vercel
   - AWS S3
   - GitHub Pages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Note**: This frontend is designed to work with a brainwave-based blockchain system. Ensure your backend is properly configured to provide the expected data format and API endpoints.
