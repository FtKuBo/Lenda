import React from 'react';
import styled from 'styled-components';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

const ChartContainer = styled.div`
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  
  @media (max-width: 768px) {
    padding: 0.5rem;
  }
  
  @media (max-width: 480px) {
    padding: 0.4rem;
  }
`;

const ChartTitle = styled.h3`
  margin: 0 0 0.5rem 0;
  font-size: 1rem;
  font-weight: 400;
  text-align: center;
  color: #fff;
  flex-shrink: 0;
  
  @media (max-width: 768px) {
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
  }
`;

const FrequencyLegend = styled.div`
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
  flex-shrink: 0;
  
  @media (max-width: 768px) {
    gap: 0.4rem;
    margin-bottom: 0.4rem;
  }
  
  @media (max-width: 480px) {
    gap: 0.3rem;
    margin-bottom: 0.3rem;
  }
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.75rem;
  
  @media (max-width: 768px) {
    font-size: 0.7rem;
    gap: 0.25rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.65rem;
    gap: 0.2rem;
  }
`;

const LegendColor = styled.div<{ color: string }>`
  width: 8px;
  height: 8px;
  background-color: ${props => props.color};
  border-radius: 2px;
  
  @media (max-width: 480px) {
    width: 6px;
    height: 6px;
  }
`;

const FrequencyInfo = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
  gap: 0.5rem;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  flex-shrink: 0;
  
  @media (max-width: 768px) {
    grid-template-columns: repeat(auto-fit, minmax(70px, 1fr));
    gap: 0.4rem;
    padding: 0.4rem;
  }
  
  @media (max-width: 480px) {
    grid-template-columns: repeat(2, 1fr);
    gap: 0.3rem;
    padding: 0.3rem;
  }
`;

const FrequencyCard = styled.div`
  text-align: center;
  padding: 0.3rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  
  @media (max-width: 480px) {
    padding: 0.25rem;
  }
`;

const FrequencyName = styled.div`
  font-weight: 600;
  font-size: 0.7rem;
  margin-bottom: 0.15rem;
  
  @media (max-width: 768px) {
    font-size: 0.65rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.6rem;
  }
`;

const FrequencyValue = styled.div`
  font-size: 0.8rem;
  font-weight: 700;
  
  @media (max-width: 768px) {
    font-size: 0.75rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.7rem;
  }
`;

const FrequencyRange = styled.div`
  font-size: 0.6rem;
  opacity: 0.7;
  margin-top: 0.15rem;
  
  @media (max-width: 768px) {
    font-size: 0.55rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.5rem;
  }
`;

interface BrainwaveData {
  timestamp: Date;
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
}

interface BrainwaveChartProps {
  data: BrainwaveData[];
}

const BrainwaveChart: React.FC<BrainwaveChartProps> = ({ data }) => {
  // Transform data for Recharts
  const chartData = data.map(item => ({
    time: item.timestamp.toLocaleTimeString(),
    alpha: Math.round(item.alpha * 100) / 100,
    beta: Math.round(item.beta * 100) / 100,
    theta: Math.round(item.theta * 100) / 100,
    delta: Math.round(item.delta * 100) / 100,
    gamma: Math.round(item.gamma * 100) / 100,
  }));

  // Get current values for frequency cards
  const currentValues = data.length > 0 ? data[data.length - 1] : null;

  // Responsive chart height
  const getChartHeight = () => {
    if (typeof window !== 'undefined') {
      if (window.innerWidth <= 480) return 120;
      if (window.innerWidth <= 768) return 140;
      return 160;
    }
    return 160;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '8px',
          padding: '12px',
          color: 'white'
        }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ 
              margin: '4px 0', 
              color: entry.color,
              fontSize: '14px'
            }}>
              {`${entry.name}: ${entry.value} Hz`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <ChartContainer>
      <ChartTitle>Real-time EEG Frequency Bands</ChartTitle>
      
      <FrequencyLegend>
        <LegendItem>
          <LegendColor color="#8884d8" />
          <span>Alpha (8-13 Hz)</span>
        </LegendItem>
        <LegendItem>
          <LegendColor color="#82ca9d" />
          <span>Beta (13-30 Hz)</span>
        </LegendItem>
        <LegendItem>
          <LegendColor color="#ffc658" />
          <span>Theta (4-8 Hz)</span>
        </LegendItem>
        <LegendItem>
          <LegendColor color="#ff7300" />
          <span>Delta (0.5-4 Hz)</span>
        </LegendItem>
        <LegendItem>
          <LegendColor color="#ff0000" />
          <span>Gamma (30-100 Hz)</span>
        </LegendItem>
      </FrequencyLegend>

      <ResponsiveContainer width="100%" height={getChartHeight()}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis 
            dataKey="time" 
            stroke="rgba(255, 255, 255, 0.7)"
            fontSize={12}
            tick={{ fill: 'rgba(255, 255, 255, 0.7)' }}
          />
          <YAxis 
            stroke="rgba(255, 255, 255, 0.7)"
            fontSize={12}
            tick={{ fill: 'rgba(255, 255, 255, 0.7)' }}
            label={{ 
              value: 'Frequency (Hz)', 
              angle: -90, 
              position: 'insideLeft',
              style: { fill: 'rgba(255, 255, 255, 0.7)' }
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area 
            type="monotone" 
            dataKey="alpha" 
            stackId="1" 
            stroke="#8884d8" 
            fill="#8884d8" 
            fillOpacity={0.6}
            name="Alpha"
          />
          <Area 
            type="monotone" 
            dataKey="beta" 
            stackId="1" 
            stroke="#82ca9d" 
            fill="#82ca9d" 
            fillOpacity={0.6}
            name="Beta"
          />
          <Area 
            type="monotone" 
            dataKey="theta" 
            stackId="1" 
            stroke="#ffc658" 
            fill="#ffc658" 
            fillOpacity={0.6}
            name="Theta"
          />
          <Area 
            type="monotone" 
            dataKey="delta" 
            stackId="1" 
            stroke="#ff7300" 
            fill="#ff7300" 
            fillOpacity={0.6}
            name="Delta"
          />
          <Area 
            type="monotone" 
            dataKey="gamma" 
            stackId="1" 
            stroke="#ff0000" 
            fill="#ff0000" 
            fillOpacity={0.6}
            name="Gamma"
          />
        </AreaChart>
      </ResponsiveContainer>

      {currentValues && (
        <FrequencyInfo>
          <FrequencyCard>
            <FrequencyName>Alpha</FrequencyName>
            <FrequencyValue>{currentValues.alpha.toFixed(1)} Hz</FrequencyValue>
            <FrequencyRange>8-13 Hz</FrequencyRange>
          </FrequencyCard>
          <FrequencyCard>
            <FrequencyName>Beta</FrequencyName>
            <FrequencyValue>{currentValues.beta.toFixed(1)} Hz</FrequencyValue>
            <FrequencyRange>13-30 Hz</FrequencyRange>
          </FrequencyCard>
          <FrequencyCard>
            <FrequencyName>Theta</FrequencyName>
            <FrequencyValue>{currentValues.theta.toFixed(1)} Hz</FrequencyValue>
            <FrequencyRange>4-8 Hz</FrequencyRange>
          </FrequencyCard>
          <FrequencyCard>
            <FrequencyName>Delta</FrequencyName>
            <FrequencyValue>{currentValues.delta.toFixed(1)} Hz</FrequencyValue>
            <FrequencyRange>0.5-4 Hz</FrequencyRange>
          </FrequencyCard>
          <FrequencyCard>
            <FrequencyName>Gamma</FrequencyName>
            <FrequencyValue>{currentValues.gamma.toFixed(1)} Hz</FrequencyValue>
            <FrequencyRange>30-100 Hz</FrequencyRange>
          </FrequencyCard>
        </FrequencyInfo>
      )}
    </ChartContainer>
  );
};

export default BrainwaveChart; 
