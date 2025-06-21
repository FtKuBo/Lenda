import React, { useRef, useEffect, useState } from 'react';
import styled from 'styled-components';
import ForceGraph2D from 'react-force-graph-2d';

const NodeContainer = styled.div`
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const NodeInfo = styled.div`
  position: absolute;
  top: 0.5rem;
  left: 0.5rem;
  background: rgba(0, 0, 0, 0.8);
  padding: 0.5rem;
  border-radius: 6px;
  color: white;
  font-size: 0.7rem;
  z-index: 10;
  max-width: 200px;
  
  @media (max-width: 768px) {
    max-width: 180px;
    padding: 0.4rem;
    font-size: 0.65rem;
  }
  
  @media (max-width: 480px) {
    max-width: 150px;
    padding: 0.3rem;
    font-size: 0.6rem;
    top: 0.3rem;
    left: 0.3rem;
  }
`;

const NodeStats = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  flex-shrink: 0;
  
  @media (max-width: 768px) {
    gap: 0.4rem;
    padding: 0.4rem;
  }
  
  @media (max-width: 480px) {
    gap: 0.3rem;
    padding: 0.3rem;
    margin-top: 0.4rem;
  }
`;

const StatCard = styled.div`
  text-align: center;
  padding: 0.3rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  
  @media (max-width: 480px) {
    padding: 0.25rem;
  }
`;

const StatLabel = styled.div`
  font-size: 0.6rem;
  opacity: 0.7;
  margin-bottom: 0.15rem;
  
  @media (max-width: 768px) {
    font-size: 0.55rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.5rem;
  }
`;

const StatValue = styled.div`
  font-size: 0.8rem;
  font-weight: 600;
  
  @media (max-width: 768px) {
    font-size: 0.75rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.7rem;
  }
`;

const Controls = styled.div`
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  display: flex;
  gap: 0.25rem;
  z-index: 10;
  
  @media (max-width: 480px) {
    top: 0.3rem;
    right: 0.3rem;
    gap: 0.2rem;
  }
`;

const ControlButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.7rem;
  transition: all 0.2s;

  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  &:active {
    transform: scale(0.95);
  }
  
  @media (max-width: 768px) {
    padding: 0.25rem 0.5rem;
    font-size: 0.65rem;
  }
  
  @media (max-width: 480px) {
    padding: 0.2rem 0.4rem;
    font-size: 0.6rem;
  }
`;

interface Node {
  id: string;
  label: string;
  type: string;
  value: number;
  timestamp: Date;
  connections: string[];
}

interface ObsidianNodesProps {
  nodes: Node[];
}

const ObsidianNodes: React.FC<ObsidianNodesProps> = ({ nodes }) => {
  const fgRef = useRef<any>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [graphData, setGraphData] = useState<any>({ nodes: [], links: [] });

  const getNodeColor = (type: string): string => {
    const colors: { [key: string]: string } = {
      alpha: '#8884d8',
      beta: '#82ca9d',
      theta: '#ffc658',
      delta: '#ff7300',
      gamma: '#ff0000',
    };
    return colors[type] || '#ffffff';
  };

  // Transform nodes data for the force graph
  useEffect(() => {
    const graphNodes = nodes.map(node => ({
      id: node.id,
      label: node.label,
      type: node.type,
      value: node.value,
      timestamp: node.timestamp,
      val: Math.max(node.value * 2, 10), // Node size based on value
      color: getNodeColor(node.type),
    }));

    const graphLinks = nodes.flatMap(node =>
      node.connections.map(targetId => ({
        source: node.id,
        target: targetId,
        color: 'rgba(255, 255, 255, 0.3)',
        width: 1,
      }))
    );

    setGraphData({ nodes: graphNodes, links: graphLinks });
  }, [nodes]);

  const handleNodeClick = (node: any) => {
    setSelectedNode(node);
  };

  const handleBackgroundClick = () => {
    setSelectedNode(null);
  };

  const zoomToFit = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  };

  const resetView = () => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000);
      fgRef.current.zoom(1, 1000);
    }
  };

  const getNodeTypeInfo = (type: string) => {
    const typeInfo: { [key: string]: { description: string; range: string } } = {
      alpha: { description: 'Relaxed, calm state', range: '8-13 Hz' },
      beta: { description: 'Active, focused thinking', range: '13-30 Hz' },
      theta: { description: 'Deep meditation, creativity', range: '4-8 Hz' },
      delta: { description: 'Deep sleep, unconscious', range: '0.5-4 Hz' },
      gamma: { description: 'High-level processing', range: '30-100 Hz' },
    };
    return typeInfo[type] || { description: 'Unknown', range: 'N/A' };
  };

  return (
    <NodeContainer>
      <Controls>
        <ControlButton onClick={zoomToFit}>Fit View</ControlButton>
        <ControlButton onClick={resetView}>Reset</ControlButton>
      </Controls>

      {selectedNode && (
        <NodeInfo>
          <h4 style={{ margin: '0 0 0.5rem 0' }}>{selectedNode.label}</h4>
          <p style={{ margin: '0.25rem 0', fontSize: '0.85rem' }}>
            <strong>Type:</strong> {selectedNode.type.toUpperCase()}
          </p>
          <p style={{ margin: '0.25rem 0', fontSize: '0.85rem' }}>
            <strong>Value:</strong> {selectedNode.value} Hz
          </p>
          <p style={{ margin: '0.25rem 0', fontSize: '0.85rem' }}>
            <strong>Time:</strong> {selectedNode.timestamp.toLocaleTimeString()}
          </p>
          <p style={{ margin: '0.25rem 0', fontSize: '0.85rem' }}>
            <strong>Connections:</strong> {selectedNode.connections.length}
          </p>
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', opacity: 0.8 }}>
            {getNodeTypeInfo(selectedNode.type).description}
          </p>
        </NodeInfo>
      )}

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="label"
        nodeColor="color"
        nodeVal="val"
        linkColor="color"
        linkWidth="width"
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        backgroundColor="transparent"
        cooldownTicks={100}
        nodeRelSize={6}
        linkCurvature={0.1}
        d3VelocityDecay={0.3}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          const label = node.label;
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(
            node.x - bckgDimensions[0] / 2,
            node.y - bckgDimensions[1] / 2,
            bckgDimensions[0],
            bckgDimensions[1]
          );

          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = 'white';
          ctx.fillText(label, node.x, node.y);

          // Draw node circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.val / globalScale, 0, 2 * Math.PI);
          ctx.fillStyle = node.color;
          ctx.fill();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 2 / globalScale;
          ctx.stroke();
        }}
      />

      <NodeStats>
        <StatCard>
          <StatLabel>Total Nodes</StatLabel>
          <StatValue>{nodes.length}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Total Connections</StatLabel>
          <StatValue>{nodes.reduce((acc, node) => acc + node.connections.length, 0) / 2}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Latest Block</StatLabel>
          <StatValue>{nodes.length > 0 ? nodes[nodes.length - 1].id : 'N/A'}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Active Types</StatLabel>
          <StatValue>{new Set(nodes.map(n => n.type)).size}</StatValue>
        </StatCard>
      </NodeStats>
    </NodeContainer>
  );
};

export default ObsidianNodes; 
