import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import BrainwaveChart from './components/BrainwaveChart';
import ObsidianNodes from './components/ObsidianNodes';
import './App.css';

const AppContainer = styled.div`
  height: 100vh;
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
  color: white;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const Header = styled.header`
  background: rgba(0, 0, 0, 0.3);
  padding: 0.75rem 2rem;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  
  @media (max-width: 768px) {
    padding: 0.5rem 1rem;
  }
`;

const LogoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const LogoIcon = styled.img`
  height: 40px;
  width: auto;
  filter: brightness(0) invert(1);
  
  @media (max-width: 768px) {
    height: 32px;
  }
  
  @media (max-width: 480px) {
    height: 28px;
  }
`;

const LogoText = styled.img`
  height: 200px;
  width: auto;
  
  @media (max-width: 768px) {
    height: 28px;
  }
  
  @media (max-width: 480px) {
    height: 24px;
  }
`;

const Title = styled.h1`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 300;
  text-align: center;
  
  @media (max-width: 768px) {
    font-size: 1.25rem;
  }
  
  @media (max-width: 480px) {
    font-size: 1.1rem;
  }
`;

const Subtitle = styled.p`
  margin: 0.25rem 0 0 0;
  text-align: center;
  opacity: 0.8;
  font-size: 0.9rem;
  
  @media (max-width: 768px) {
    font-size: 0.8rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.75rem;
  }
`;

const StatusContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const MainContent = styled.main`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  padding: 1rem;
  flex: 1;
  min-height: 0;
  position: relative;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 0.75rem;
    padding: 0.75rem;
  }
  
  @media (max-width: 480px) {
    padding: 0.5rem;
    gap: 0.5rem;
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 200px;
    height: 200px;
    background-image: url('/lendaLogo1.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.03;
    pointer-events: none;
    z-index: 0;
    
    @media (max-width: 768px) {
      width: 150px;
      height: 150px;
    }
    
    @media (max-width: 480px) {
      width: 100px;
      height: 100px;
    }
  }
`;

const Section = styled.section`
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  z-index: 1;
  
  @media (max-width: 768px) {
    padding: 0.75rem;
    border-radius: 10px;
  }
  
  @media (max-width: 480px) {
    padding: 0.5rem;
    border-radius: 8px;
  }
`;

const SectionTitle = styled.h2`
  margin: 0 0 0.75rem 0;
  font-size: 1.25rem;
  font-weight: 400;
  color: #fff;
  text-align: center;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  
  @media (max-width: 768px) {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
  }
  
  @media (max-width: 480px) {
    font-size: 1rem;
    margin-bottom: 0.4rem;
  }
`;

const StatusIndicator = styled.div<{ isConnected: boolean }>`
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: ${props => props.isConnected ? '#4CAF50' : '#f44336'};
  margin-right: 6px;
  animation: ${props => props.isConnected ? 'pulse 2s infinite' : 'none'};
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
  
  @media (max-width: 480px) {
    width: 8px;
    height: 8px;
    margin-right: 4px;
  }
`;

const StatusText = styled.span`
  font-size: 0.8rem;
  opacity: 0.9;
  
  @media (max-width: 480px) {
    font-size: 0.75rem;
  }
`;

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [brainwaveData, setBrainwaveData] = useState<any[]>([]);
  const [nodes, setNodes] = useState<any[]>([]);

  // Simulate real-time brainwave data
  useEffect(() => {
    const generateBrainwaveData = () => {
      const now = new Date();
      const data = [];
      
      for (let i = 0; i < 100; i++) {
        const time = new Date(now.getTime() - (100 - i) * 100);
        data.push({
          timestamp: time,
          alpha: Math.random() * 20 + 10,
          beta: Math.random() * 15 + 5,
          theta: Math.random() * 25 + 15,
          delta: Math.random() * 30 + 20,
          gamma: Math.random() * 10 + 5
        });
      }
      
      setBrainwaveData(data);
    };

    // Generate initial data
    generateBrainwaveData();
    
    // Simulate connection
    setTimeout(() => setIsConnected(true), 2000);
    
    // Update data every 2 seconds
    const interval = setInterval(generateBrainwaveData, 2000);
    
    return () => clearInterval(interval);
  }, []);

  // Generate sample blockchain nodes
  useEffect(() => {
    const generateNodes = () => {
      const sampleNodes = [
        {
          id: '1',
          label: 'Brainwave Block #1',
          type: 'alpha',
          value: 15.2,
          timestamp: new Date(Date.now() - 60000),
          connections: ['2', '3']
        },
        {
          id: '2',
          label: 'Brainwave Block #2',
          type: 'beta',
          value: 8.7,
          timestamp: new Date(Date.now() - 30000),
          connections: ['1', '4']
        },
        {
          id: '3',
          label: 'Brainwave Block #3',
          type: 'theta',
          value: 22.1,
          timestamp: new Date(Date.now() - 15000),
          connections: ['1', '5']
        },
        {
          id: '4',
          label: 'Brainwave Block #4',
          type: 'delta',
          value: 28.5,
          timestamp: new Date(Date.now() - 5000),
          connections: ['2', '6']
        },
        {
          id: '5',
          label: 'Brainwave Block #5',
          type: 'gamma',
          value: 12.3,
          timestamp: new Date(),
          connections: ['3', '6']
        },
        {
          id: '6',
          label: 'Brainwave Block #6',
          type: 'alpha',
          value: 16.8,
          timestamp: new Date(),
          connections: ['4', '5']
        }
      ];
      
      setNodes(sampleNodes);
    };

    generateNodes();
  }, []);

  return (
    <AppContainer>
      <Header>
        <LogoContainer>
          <LogoText src="/lendaLogo2.png" alt="Lenda Logo Text" />
        </LogoContainer>
        <div>
          <Title>Lenda Biochain</Title>
          <Subtitle>
            <StatusContainer>
              <StatusIndicator isConnected={isConnected} />
              <StatusText>
                {isConnected ? 'Connected to Raspberry Pi & EEG Device' : 'Connecting to devices...'}
              </StatusText>
            </StatusContainer>
          </Subtitle>
        </div>
      </Header>

      <MainContent>
        <Section>
          <SectionTitle>Real-time Brainwave Analysis</SectionTitle>
          <BrainwaveChart data={brainwaveData} />
        </Section>

        <Section>
          <SectionTitle>Blockchain Node Network</SectionTitle>
          <ObsidianNodes nodes={nodes} />
        </Section>
      </MainContent>
    </AppContainer>
  );
}

export default App;
