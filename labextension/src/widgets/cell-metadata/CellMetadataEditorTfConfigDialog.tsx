/*
 * Copyright 2020 The Kale Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as React from 'react';
import {
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Switch,
} from '@material-ui/core';
import ColorUtils from '../../lib/ColorUtils';
import { Input } from '../../components/Input';
import { Select } from '../../components/Select';

interface ICellMetadataEditorTfConfigDialog {
  open: boolean;
  stepName: string;
  distribute: string;
  updateDistribute: Function;
  numWorkers: string;
  numParameterServers: string;
  toggleDialog: Function;
}

export const CellMetadataEditorTfConfigDialog: React.FunctionComponent<ICellMetadataEditorTfConfigDialog> = props => {
  const handleClose = () => {
    props.toggleDialog();
  };

  const distributeAction = (
    action: string,
    distributeKey: string,
    distributeValue: string,
  ) => {
    return {
      action,
      distributeKey,
      distributeValue,
    };
  };

  const distribute = props.distribute !== '' ? props.distribute.split(':')[1] : '';
  const numWorkers = props.numWorkers !== '' ? props.numWorkers.split(':')[1] : '';

  return (
    <Dialog
      open={props.open}
      onClose={handleClose}
      fullWidth={true}
      maxWidth={'sm'}
      scroll="paper"
      aria-labelledby="scroll-dialog-title"
      aria-describedby="scroll-dialog-description"
    >
      <DialogTitle id="scroll-dialog-title">
        <Grid
          container
          direction="row"
          justify="space-between"
          alignItems="center"
        >
          <Grid item xs={9}>
            <Grid
              container
              direction="row"
              justify="flex-start"
              alignItems="center"
            >
              <p>Set TF_CONFIG for step </p>
              <Chip
                className={'kale-chip'}
                style={{
                  marginLeft: '10px',
                  backgroundColor: `#${ColorUtils.getColor(props.stepName)}`,
                }}
                key={props.stepName}
                label={props.stepName}
              />
            </Grid>
          </Grid>
          <Grid item xs={3}>
            <Grid
              container
              direction="row"
              justify="flex-end"
              alignItems="center"
            >
              <Switch
                checked={distribute !== ''}
                onChange={c => {
                  if (c.target.checked) {
                    // default value
                    props.updateDistribute([
                      distributeAction('update', 'distribute', 'MultiWorkerMirroredStrategy'),
                      distributeAction('update', 'numWorkers', '1'),
                    ]);
                  } else {
                    props.updateDistribute([
                      distributeAction('delete', '','')
                    ]);
                  }
                }}
                color="primary"
                name="enableKale"
                inputProps={{ 'aria-label': 'primary checkbox' }}
                classes={{ root: 'material-switch' }}
              />
            </Grid>
          </Grid>
        </Grid>
      </DialogTitle>
      <DialogContent dividers={true} style={{ paddingTop: 0 }}>
        <Grid container direction="column" justify="center" alignItems="center">
          <Grid
            container
            direction="column"
            justify="space-between"
            alignItems="center"
          >
            <Grid
              container
              direction="row"
              justify="space-between"
              alignItems="center"
              style={{ marginTop: '15px' }}
            >
              <Grid item xs={6}>
                <Select
                  disabled={distribute === ''}
                  variant="standard"
                  label="Strategy"
                  values={[
                    {label:'MultiWorkerMirroredStrategy', value:'MultiWorkerMirroredStrategy'},
                    // {label:'ParameterServerStrategy', value:'ParameterServerStrategy'}
                  ]}
                  value={distribute === '' ? 'MultiWorkerMirroredStrategy' : distribute === 'MultiWorkerMirroredStrategy' ? 'MultiWorkerMirroredStrategy' : distribute === 'ParameterServerStrategy' ? 'ParameterServerStrategy' : null}
                  index={0}
                  updateValue={()=>{}}
                  style={{ width: 'auto', float: 'left' }}
                />
              </Grid>
            </Grid>
            <Grid
              container
              direction="row"
              justify="space-between"
              alignItems="center"
              style={{ marginTop: '15px' }}
            >
              <Grid item xs={6}>
                <Input
                  disabled={numWorkers === ''}
                  variant="outlined"
                  label="Number of Workers"
                  value={parseInt(numWorkers) || 1}
                  updateValue={(v: number) =>
                    props.updateDistribute([
                      distributeAction('update', 'numWorkers', v.toString()),
                    ])
                  }
                  style={{ width: '95%', float: 'left' }}
                />
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} color="primary">
          Ok
        </Button>
      </DialogActions>
    </Dialog>
  );
};
