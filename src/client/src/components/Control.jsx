import React, { Fragment, useEffect, useState } from 'react'
import {
    Grid, Paper, Typography,
    FormControl, InputLabel, FormHelperText, Button,
} from '@material-ui/core';
import { useDispatch, useSelector } from 'react-redux';
import useCollapse from 'react-collapsed';


function ControlWrapper() {
    const { getCollapseProps, getToggleProps, isExpanded } = useCollapse();

    return (
        <Fragment>
            <div className="collapsible content-container-fit" {...getToggleProps()}>
                <Button justifyContent="flex-end">
                    <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                        Settings
                    </Typography>
                </Button>
                {isExpanded ? (
                    <Paper className='content-container-fit' elevation={3}>
                        <Grid container item direction='column' spacing={2}>
                            <Controller />
                        </Grid>
                        <Grid container item direction='column' spacing={2}>
                        </Grid>
                    </Paper>
                ): (
                    <div {...getCollapseProps()}>
                    </div>
                )}
            </div>
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
