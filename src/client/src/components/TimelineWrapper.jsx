import React, { Fragment } from 'react'
import {
    Grid, Box, Typography,
} from '@material-ui/core';


function TimelineWrapper() {
    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Execution Timeline
            </Typography>
            <Grid item>
                <Timeline />
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function Timeline() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default TimelineWrapper;
