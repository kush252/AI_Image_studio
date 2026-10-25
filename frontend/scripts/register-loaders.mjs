import { register } from 'node:module'
import { pathToFileURL } from 'node:url'

register('./css-stub-loader.mjs', pathToFileURL('./scripts/'))
