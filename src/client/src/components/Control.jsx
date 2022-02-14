import React, { Fragment, useEffect, useState } from 'react'
import {
    Grid, Paper, Typography,
    FormControl, InputLabel, FormHelperText,
} from '@material-ui/core';
import { useDispatch, useSelector } from 'react-redux';

function ControlWrapper() {
    return (
        <Fragment>
            <Paper className='content-container-fit'>
                <Grid item>
                    <Typography variant='overline' style={{ fontWeight: 'bold' }}>Settings</Typography>
                </Grid>
                <Grid container item direction='column' spacing={2}>
                    <Controller />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>

            </Paper>
        </Fragment>
    )
}

function Controller() {
    const dispatch = useDispatch();
    const datasets = useSelector(store => store.datasets);
    const [selectedDataset, setSelectedDataset] = useState('');

    useEffect(() => {
        if (datasets.length > 0) {
            // setSelectedDataset(datasets[0]);
            // dispatch(fetchCA(datasets[0]));
        }
    }, [datasets])

    useEffect(() => {
        // dispatch(fetchDatasets("CachedArray"));
    }, []);

    useEffect(() => {
        // dispatch(fetchCA(selectedDataset));
    }, [selectedDataset]);

    return (
        <Fragment>
            <Grid item>
                { datasets.length > 0 ? (
                    <FormControl className='form-control-fit'>
                        <InputLabel id="dataset-label">Dataset</InputLabel>
                            {/* <Select labelId="dataset-label" id="dataset-select"
                            value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)} >
                            { datasets.map((cc) => (
                                <MenuItem key={cc} value={cc}>{cc}</MenuItem>
                            )) }
                            </Select> */}
                            <FormHelperText>Select the run</FormHelperText>
                    </FormControl>
                ): (<></>)} 
            </Grid>
        </Fragment>
    )
}


export default ControlWrapper;
