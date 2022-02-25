import React, { Fragment } from 'react'
import {
    Grid, Box, Typography,
} from '@material-ui/core';


function CCTWrapper() {
    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Calling Context Tree
            </Typography>
            <Grid item>
                <CCT />
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function CCT() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default CCTWrapper;
