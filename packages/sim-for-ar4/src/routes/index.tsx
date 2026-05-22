import { createFileRoute } from '@tanstack/react-router';
import Ar4Simulator from '../components/Ar4Simulator';

export const Route = createFileRoute('/')({
  component: RouteComponent,
});

function RouteComponent() {
  return (
    <div style={{ height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      <Ar4Simulator />
    </div>
  );
}

