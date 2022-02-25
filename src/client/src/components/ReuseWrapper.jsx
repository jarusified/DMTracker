import React, { Fragment } from 'react'
import {
    Grid, Box, Typography,
} from '@material-ui/core';


function ReuseWrapper() {
    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Data Reuse analysis
            </Typography>
            <Grid item>
                <Encoding />
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function Encoding() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default ReuseWrapper;
