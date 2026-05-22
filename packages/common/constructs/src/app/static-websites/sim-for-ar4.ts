import * as url from 'url';
import { Construct } from 'constructs';
import { StaticWebsite } from '../../core/index.js';

export class SimForAr4 extends StaticWebsite {
  constructor(scope: Construct, id: string) {
    super(scope, id, {
      websiteName: 'SimForAr4',
      websiteFilePath: url.fileURLToPath(
        new URL(
          '../../../../../../dist/packages/sim-for-ar4/bundle',
          import.meta.url,
        ),
      ),
    });
  }
}
