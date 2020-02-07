import React from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import Slider, { Range } from 'rc-slider';
import sma from 'sma';
import { Line } from '@nivo/line';
import './App.css';
import 'rc-slider/assets/index.css';

function App() {

  const initialData = {
    lineChart: {}
  };

  const [chartData, setChartData] = React.useState(initialData);
  const [messageCount, setMessageCount] = React.useState(0);
  const [movingAverage, setMovingAverage] = React.useState(0);
  const [sendMessage, lastMessage, readyState, getWebSocket] = useWebSocket("ws://localhost:8080");

  React.useEffect(() => {
      if(lastMessage !== null) {
        const data = JSON.parse(lastMessage.data);
        // console.log(data);
        const id   = data.model_name;
        const y    = data.loss;
        const x    = messageCount;
        setChartData({
          ...chartData,
          lineChart: {
            ...chartData.lineChart,
            [id]: (chartData.lineChart[id] || []).concat({ x, y })
          }
        });
        // increment the count.
        setMessageCount(messageCount + 1);
      }
  }, [lastMessage]);


  const finalChartData = Object.keys(chartData.lineChart).map(id => {
    
    if (movingAverage === 0) {
      return {
        id,
        data: chartData.lineChart[id]
      }
    } else {
      const series = chartData.lineChart[id].map(({y}) => y);
      const windowLength = Math.ceil(series.length * movingAverage * 0.6);
      const averagedData = sma(series, windowLength);
      return {
        id,
        data: averagedData.map((y, i) => ({ y, x: chartData.lineChart[id][i].x}))
      }
    }
  });

  console.log(finalChartData);
  return (
    <div className="App">
      <div className="App-Row Header">
        Remote Monitor
        <button onClick={() => setChartData(initialData)}>
          Reset
        </button>
      </div>
      <div className="App-Row">
        <div style={{flex: 1}}>
          <Line data={finalChartData}
            height={600}
            width={800}
            margin={{ top: 50, right: 50, bottom: 50, left: 60 }}
            enablePoints={false}
            enableArea={true}
            enableCrosshair={true}
            axisBottom={{
              tickValues: 4,
            }}
            axisLeft={{
                tickValues: 4,
            }}
            xScale={{
              type: 'linear'
            }}
            yScale={{
              type: 'linear',
              stacked: true
            }}
          />
        </div>
        <div style={{flex: 1}}>
            
          <div>
            Moving Average: {movingAverage}
            <Slider max={1} min={0} step={0.05} onAfterChange={v => setMovingAverage(v)} />
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;
