import React, { Fragment } from 'react'
import {
    Grid, Box, Typography,
} from '@material-ui/core';


function MetricsWrapper() {
    return ( 
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Performance metrics
            </Typography>
            <Grid item>
                <Metrics />
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function Metrics() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default MetricsWrapper;
